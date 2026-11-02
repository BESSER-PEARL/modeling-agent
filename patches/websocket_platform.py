from __future__ import annotations

import base64
import inspect
import json
import os
import time
import traceback
from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import numpy as np
import subprocess
import threading
from typing import TYPE_CHECKING

from pandas import DataFrame
from websockets.exceptions import ConnectionClosedError
from websockets.sync.server import ServerConnection, WebSocketServer, serve

from baf.core.gui.agent_gui import AgentGUI
from baf.core.gui.gui_serializer import gui_to_json
from baf.library.transition.events.base_events import ReceiveMessageEvent, ReceiveFileEvent, GUIEvent
from baf.core.message import Message, MessageType
from baf.core.session import Session
from baf.exceptions.exceptions import PlatformMismatchError, StreamlitDatabaseException
from baf.exceptions.logger import logger
from baf.nlp.rag.rag import RAGMessage
from baf.platforms import websocket
from baf.platforms.payload import Payload, PayloadAction, PayloadEncoder
from baf.platforms.platform import Platform
from baf.platforms.websocket.streamlit_ui import streamlit_ui
from baf.core.file import File
from baf.platforms.websocket.streamlit_ui import (
    DB_STREAMLIT_HOST,
    DB_STREAMLIT_PORT,
    DB_STREAMLIT_DATABASE,
    DB_STREAMLIT_USERNAME,
    DB_STREAMLIT_PASSWORD,
    DB_STREAMLIT,
)


# Session-variable keys whose values must never be written to the logs.
# The frontend sends the user's BYOK key as ``user_api_key`` via
# USER_SET_VARIABLE; redact it (and any other obviously-sensitive key)
# rather than logging the value in plaintext.
_SENSITIVE_VAR_KEYS = {
    "user_api_key", "api_key", "apikey", "secret", "password", "token", "auth",
}


def _set_request_byok(session):
    """Set the per-request BYOK context var from the session's stored key.

    The frontend sends the user's key/provider/model via USER_SET_VARIABLE,
    so they live on the session under ``user_api_key`` / ``user_api_provider``
    / ``user_api_model``. ``byok`` is imported lazily because this file is
    copied into the installed ``baf`` package at build time; ``src/`` is on
    ``sys.path`` once the agent has started (and the handler only runs then).

    Returns a token to pass to :func:`_reset_request_byok`, or ``None`` if
    BYOK is unavailable for any reason (the request then uses the shared LLM).
    """
    try:
        import byok
        return byok.set_current(
            session.get("user_api_provider"),
            session.get("user_api_key"),
            session.get("user_api_model"),
            session.get("user_api_base"),
        )
    except Exception as exc:  # never let BYOK wiring break message handling
        logger.warning(f"BYOK: could not set per-request key: {exc}")
        return None


def _reset_request_byok(token) -> None:
    """Reset the per-request BYOK context var (best effort)."""
    if token is None:
        return
    try:
        import byok
        byok.reset_current(token)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"BYOK: could not reset per-request key: {exc}")


def _tab_id_of(payload_str) -> str | None:
    """The sending tab's id: the editor puts its per-tab session id in the raw
    payload's ``user_id``, which ``Payload.decode`` drops."""
    try:
        data = json.loads(payload_str) if isinstance(payload_str, (str, bytes)) else payload_str
    except (TypeError, ValueError):
        return None
    value = data.get("user_id") if isinstance(data, dict) else None
    return str(value) if value else None


if TYPE_CHECKING:
    from baf.core.agent import Agent

try:
    import cv2
except ImportError:
    logger.warning("cv2 dependencies in WebSocketPlatform could not be imported. You can install them from "
                   "the requirements/requirements-extras.txt file")
try:
    import plotly
except ImportError:
    logger.warning("plotly dependencies in WebSocketPlatform could not be imported. You can install them from "
                   "the requirements/requirements-extras.txt file")

try:
    import librosa
except ImportError:
    logger.warning("librosa dependencies in WebSocketPlatform could not be imported. You can install them from "
                   "the requirements/requirements-extras.txt file")


def _extract_parameter_from_request(parameter, request) -> str | None:
    if not request:
        return None
    for attr in ("path", "raw_path", "uri"):
        value = getattr(request, attr, None)
        if not value:
            continue
        if isinstance(value, bytes):
            # Prefer UTF-8 for URL paths; fall back to latin-1 to remain robust to non-UTF-8 bytes.
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.decode("latin-1", errors="replace")
        query = urlsplit(value).query
        if not query:
            continue
        params = parse_qs(query)
        user_values = params.get(parameter)
        if user_values:
            return user_values[0]
    return None


class WebSocketPlatform(Platform):
    """The WebSocket Platform allows an agent to communicate with the users using the
    `WebSocket <https://en.wikipedia.org/wiki/WebSocket>`_ bidirectional communications protocol.

    This platform implements the WebSocket server, and it can establish connection with a client, allowing the
    bidirectional communication between server and client (i.e. sending and receiving messages).

    Note:
        We provide different interfaces implementing a WebSocket client to communicate with the agent, though you
        can use or create your own UI as long as it has a WebSocket client that connects to the agent's WebSocket server.

    Args:
        agent (Agent): the agent the platform belongs to
        use_ui (bool): whether to use the built-in UI or not
        authenticate_users (bool): whether to authenticate users when they connect to the agent and load their chat 
            history

    Attributes:
        _agent (Agent): The agent the platform belongs to
        _host (str): The WebSocket host address (e.g. `localhost`)
        _port (int): The WebSocket port (e.g. `8765`)
        _use_ui (bool): Whether to use the built-in UI or not
        _connections (dict[str, ServerConnection]): The list of active connections (i.e. users connected to the agent)
        _websocket_server (WebSocketServer or None): The WebSocket server instance
        _message_handler (Callable[[ServerConnection], None]): The function that handles the user connections
            (sessions) and incoming messages
    """

    def __init__(
            self,
            agent: 'Agent',
            use_ui: bool = True,
            authenticate_users: bool = False,
    ):
        super().__init__()
        self._agent: 'Agent' = agent
        self._host: str = None
        self._port: int = None
        self._use_ui: bool = use_ui
        self._authenticate_users = authenticate_users
        self._connections: dict[str, ServerConnection] = {}
        # Browser tab (the payload's per-tab ``user_id``) whose connection holds
        # each session's reply slot. Every tab of one browser shares the session
        # key (the stable ``?user_id=``), so without this a heartbeat from another
        # open editor tab steals the slot and the in-flight reply goes there.
        self._slot_tab: dict[str, str | None] = {}
        # Per-session outbox: replies that can't be delivered right now (the
        # slot holds a stale/half-open socket, or the client is mid-reconnect)
        # are buffered here and flushed on the next slot reclaim (USER_MESSAGE
        # / __heartbeat), so a fast reply produced during a reconnect gap is
        # delivered late instead of lost. Bounded per session.
        self._outbox: dict[str, list[Payload]] = {}
        self._outbox_lock = threading.Lock()
        self._outbox_max = 50
        self._websocket_server: WebSocketServer = None

        def message_handler(conn: ServerConnection) -> None:
            """This method is run on each user connection to handle incoming messages and the agent sessions.

            Args:
                conn (ServerConnection): the user connection
            """
            session: Session = None
            current_time = datetime.now()
            request = getattr(conn, "request", None)
            headers = getattr(request, "headers", {}) if request else {}
            header_user = headers.get("X-User-ID") if hasattr(headers, "get") else None
            header_session = headers.get("X-Session-ID") if hasattr(headers, "get") else None
            query_user = _extract_parameter_from_request('user_id', request)
            query_session = _extract_parameter_from_request('session_id', request)
            query_session_name = _extract_parameter_from_request('session_name', request)
            username = header_user or query_user
            # The editor sends only the stable ?user_id=, so it keys the session;
            # stock BAF falls back to conn.id, a new session on every reconnect.
            session_key = header_session or query_session or username or str(conn.id)
            # Take only an empty slot: a new socket does not yet know its tab, and
            # another tab may have a reply in flight. Its first heartbeat (sent on
            # connect) claims the slot when it is this tab's.
            if str(session_key) not in self._connections:
                self._claim_slot(str(session_key), conn, None, sender=True)
            session = self._agent.get_or_create_session(session_key, self, username, query_session_name)
            try:

                for payload_str in conn:
                    if not self.running:
                        raise ConnectionClosedError(None, None)
                    payload: Payload = Payload.decode(payload_str)
                    tab_id = _tab_id_of(payload_str)

                    if payload.action == PayloadAction.FETCH_USER_MESSAGES.value:
                        try:
                            chat_history = session.get_chat_history(until_timestamp=current_time)
                            reasoning_events = session.get_reasoning_events(until_timestamp=current_time)

                            # Merge both streams in chronological order so the
                            # UI's existing aggregation (reasoning_started /
                            # reasoning_finished brackets, task panel updates)
                            # rebuilds the trace exactly as it was streamed.
                            timeline: list[tuple[datetime, str, object]] = []
                            for message in chat_history:
                                action = message.get_action()
                                if action is None:
                                    # Defensive: skip rows whose MessageType
                                    # has no PayloadAction mapping (would
                                    # crash on .value below).
                                    continue
                                timeline.append((message.timestamp, 'chat',
                                                 (action, message.content, message.timestamp)))
                            for ev in reasoning_events:
                                if ev['kind'] == 'reasoning_step':
                                    action = PayloadAction.AGENT_REPLY_REASONING_STEP
                                elif ev['kind'] == 'task_list_update':
                                    action = PayloadAction.AGENT_REPLY_TASK_LIST_UPDATE
                                else:
                                    continue
                                timeline.append((ev['timestamp'], 'reasoning',
                                                 (action, ev['payload'], ev['timestamp'])))

                            timeline.sort(key=lambda t: t[0])
                            for _, _, item in timeline:
                                action, message_content, ts = item
                                history_payload = Payload(action=action,
                                                          message=message_content,
                                                          history=True,
                                                          timestamp=ts)
                                self._send(session.id, history_payload)
                        except Exception as e:
                            logger.error(f"Error fetching chat history: {e}")
                    elif payload.action == PayloadAction.USER_MESSAGE.value:
                        # Route replies back to whichever conn sent this message
                        self._claim_slot(str(session_key), conn, tab_id, sender=True)
                        logger.info("REPLY_ROUTE: key=%r -> conn=%r (message sender)" % (session_key, conn.id))
                        # Deliver anything buffered during a prior reconnect gap.
                        self._flush_outbox(str(session_key), conn)
                        event: ReceiveMessageEvent = ReceiveMessageEvent.create_event_from(
                            message=payload.message,
                            session=session,
                            human=True)
                        # BYOK request boundary: if the user supplied their own
                        # API key (stored on the session by USER_SET_VARIABLE),
                        # set the per-request context var BEFORE receive_event.
                        # receive_event -> Session.call_manage_transition ->
                        # loop.call_soon_threadsafe captures a copy of THIS
                        # thread's context, so the value propagates into the
                        # session's event-loop thread for this turn. Reset after
                        # so the handler thread never leaks the key.
                        _byok_token = _set_request_byok(session)
                        try:
                            self._agent.receive_event(event)
                        finally:
                            _reset_request_byok(_byok_token)
                    elif payload.action == PayloadAction.USER_VOICE.value:
                        # Decode the base64 string to get audio bytes
                        audio_bytes = base64.b64decode(payload.message.encode('utf-8'))
                        message = self._agent.nlp_engine.speech2text(session, audio_bytes)

                        # Send transcribed message back to the client
                        payload = Payload(action=PayloadAction.USER_MESSAGE, message=message, history=True, timestamp=datetime.now())
                        self._send(session.id, payload)

                        event: ReceiveMessageEvent = ReceiveMessageEvent.create_event_from(
                            message=message,
                            session=session,
                            human=True)
                        # BYOK request boundary (voice turn) — see USER_MESSAGE.
                        _byok_token = _set_request_byok(session)
                        try:
                            self._agent.receive_event(event)
                        finally:
                            _reset_request_byok(_byok_token)
                    elif payload.action == PayloadAction.USER_FILE.value:
                        event: ReceiveFileEvent = ReceiveFileEvent(
                            file=File.decode(payload.message),
                            session_id=session.id,
                            human=True)
                        self._agent.receive_event(event)
                    elif payload.action == PayloadAction.AGENT_REPLY_STR.value:
                        event: ReceiveMessageEvent = ReceiveMessageEvent.create_event_from(
                            message=payload.message,
                            session=session,
                            human=False)
                        self._agent.receive_event(event)
                    elif payload.action == PayloadAction.RESET.value:
                        self._agent.reset(session.id)
                    elif payload.action == PayloadAction.USER_SET_VARIABLE.value:
                        if not isinstance(payload.message, dict) or not payload.message:
                            logger.error('Invalid message format for USER_SET_VARIABLE')
                            continue  # skip this iteration
                        # Client keep-alive heartbeat. Re-claim the reply slot for
                        # this (demonstrably live) connection so a reply produced
                        # during a long generation routes back to whichever socket
                        # the client currently holds -- even after reconnects. Then
                        # skip without setting a variable or spamming the logs.
                        if '__heartbeat' in payload.message:
                            if self._claim_slot(str(session_key), conn, tab_id, sender=False):
                                # A heartbeat proves this socket is live -> flush any
                                # replies buffered while the slot was stale/absent.
                                self._flush_outbox(str(session_key), conn)
                            continue
                        for key, value in payload.message.items():
                            session.set(key, value)
                            # Never log sensitive values (e.g. the user's BYOK
                            # API key arrives here as ``user_api_key``).
                            safe_value = (
                                "(redacted)"
                                if str(key).lower() in _SENSITIVE_VAR_KEYS
                                else value
                            )
                            logger.info(f"Session variable {key} set to {safe_value}.")
                    elif payload.action == PayloadAction.USER_GUI_EVENT.value:
                        event_data = payload.message
                        if isinstance(event_data, str):
                            try:
                                event_data = json.loads(event_data)
                            except Exception:
                                event_data = {'raw': event_data}
                        if not isinstance(event_data, dict):
                            event_data = {}
                        event: GUIEvent = GUIEvent(
                            event_data=event_data,
                            session_id=session.id,
                            message_id=event_data.get('messageId'))
                        self._agent.receive_event(event)
            except ConnectionClosedError:
                logger.info('Client connection closed unexpectedly')
            except Exception as e:
                logger.error(f"Server Error: {e}")
                traceback.print_exc()
            finally:
                # Only remove the slot if this conn still owns it
                if session:
                    session_id = str(session.id)
                    if session_id in self._connections and self._connections[session_id] is conn:
                        del self._connections[session_id]
                        self._slot_tab.pop(session_id, None)
                logger.info('Session finished')
                # self._agent.delete_session(session.id)
                # del self._connections[session.id]

        self._message_handler = message_handler

    def initialize(self) -> None:
        self._host = self._agent.get_property(websocket.WEBSOCKET_HOST)
        self._port = self._agent.get_property(websocket.WEBSOCKET_PORT)
        origins = self._agent.get_property(websocket.WEBSOCKET_ORIGINS)
        self._websocket_server = serve(
            handler=self._message_handler,
            host=self._host,
            port=self._port,
            max_size=self._agent.get_property(websocket.WEBSOCKET_MAX_SIZE),
            origins=origins,
        )

    def start(self) -> None:
        if self._use_ui:
            def run_streamlit() -> None:
                """Run the Streamlit UI in a dedicated thread."""
                if self._authenticate_users:
                    db_host = self._agent.get_property(DB_STREAMLIT_HOST)
                    if not db_host:
                        raise StreamlitDatabaseException("DB_STREAMLIT_HOST")
                    db_port = self._agent.get_property(DB_STREAMLIT_PORT)
                    if not db_port:
                        raise StreamlitDatabaseException("DB_STREAMLIT_PORT")
                    db_name = self._agent.get_property(DB_STREAMLIT_DATABASE)
                    if not db_name:
                        raise StreamlitDatabaseException("DB_STREAMLIT_DATABASE")
                    db_user = self._agent.get_property(DB_STREAMLIT_USERNAME)
                    if not db_user:
                        raise StreamlitDatabaseException("DB_STREAMLIT_USERNAME")
                    db_password = self._agent.get_property(DB_STREAMLIT_PASSWORD)
                    if not db_password:
                        raise StreamlitDatabaseException("DB_STREAMLIT_PASSWORD")
                    db_streamlit = self._agent.get_property(DB_STREAMLIT)
                    if not db_streamlit:
                        raise StreamlitDatabaseException("DB_STREAMLIT")

                    os.environ["STREAMLIT_DB_HOST"] = str(db_host) if db_host else ""
                    os.environ["STREAMLIT_DB_PORT"] = str(db_port) if db_port else ""
                    os.environ["STREAMLIT_DB_NAME"] = str(db_name) if db_name else ""
                    os.environ["STREAMLIT_DB_USER"] = str(db_user) if db_user else ""
                    os.environ["STREAMLIT_DB_PASSWORD"] = str(db_password) if db_password else ""
                    os.environ["STREAMLIT_DB"] = str(db_streamlit) if db_streamlit else "False"

                # Pass user profiles to Streamlit via environment, if available and serializable
                if self._agent.user_profiles is not None:
                    try:
                        os.environ["STREAMLIT_USER_PROFILES_JSON"] = json.dumps(self._agent.user_profiles)
                    except (TypeError, ValueError):
                        logger.warning("Failed to serialize user profiles for Streamlit UI; continuing without them.")

                os.environ["STREAMLIT_AGENT_WORKDIR"] = os.getcwd()

                default_chat_style = {
                    "size": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_SIZE),
                    "font": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_FONT),
                    "lineSpacing": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_LINE_SPACING),
                    "alignment": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_ALIGNMENT),
                    "color": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_COLOR),
                    "contrast": self._agent.get_property(websocket.STREAMLIT_CHAT_DEFAULT_CONTRAST),
                }
                os.environ["STREAMLIT_CHAT_INTERFACE_DEFAULT_STYLE_JSON"] = json.dumps(default_chat_style)

                try:
                    raw_configurations = self._agent.agent_configurations

                    if not raw_configurations:
                        import __main__  # noqa: PLC0415
                        main_configurations = getattr(__main__, "agent_configurations", None)
                        if isinstance(main_configurations, dict) and main_configurations:
                            raw_configurations = main_configurations
                            self._agent.set_agent_configurations(main_configurations)

                    if isinstance(raw_configurations, dict) and raw_configurations:
                        os.environ["STREAMLIT_AGENT_CONFIGURATIONS_JSON"] = json.dumps(raw_configurations)
                    else:
                        os.environ.pop("STREAMLIT_AGENT_CONFIGURATIONS_JSON", None)
                except Exception:
                    logger.warning("Failed to serialize agent configurations for Streamlit UI; using defaults.")

                subprocess.run([
                    "streamlit", "run",
                    "--server.address", self._agent.get_property(websocket.STREAMLIT_HOST),
                    "--server.port", str(self._agent.get_property(websocket.STREAMLIT_PORT)),
                    os.path.abspath(inspect.getfile(streamlit_ui)),
                    self._agent.name,
                    self._agent.get_property(websocket.WEBSOCKET_HOST),
                    str(self._agent.get_property(websocket.WEBSOCKET_PORT))
                ])

            thread = threading.Thread(target=run_streamlit)
            logger.info('Running Streamlit UI in another thread')
            thread.start()
            # To avoid re-running the streamlit process, set self._use_ui to False
            self._use_ui = False
        logger.info(f'{self._agent.name}\'s WebSocketPlatform starting at ws://{self._host}:{self._port}')
        self.running = True
        self._websocket_server.serve_forever()

    def stop(self):
        self.running = False
        for conn_id in list(self._connections.keys()):
            conn = self._connections[conn_id]
            conn.close_socket()
        try:
            while self._connections:
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.warning('Interrupted while waiting for WebSocket connections to close; continuing shutdown.')
        self._websocket_server.shutdown()
        logger.info(f'{self._agent.name}\'s WebSocketPlatform stopped')

    def _send(self, session_id, payload: Payload) -> None:
        session_id = str(session_id)
        conn = self._connections.get(session_id)
        if conn is not None:
            try:
                conn.send(json.dumps(payload, cls=PayloadEncoder))
                logger.info(
                    "REPLY_SENT: session=%r action=%s -> conn=%r",
                    session_id, getattr(payload, "action", "?"), getattr(conn, "id", "?"),
                )
                return
            except Exception as e:
                # ``conn.send`` on a stale/half-open socket can raise (and on a
                # half-open TCP may not) -- either way the slot is bad. Prune it
                # and fall through to buffering so the reply is redelivered when
                # the client's next message/heartbeat reclaims a live slot.
                logger.warning(
                    "REPLY_SEND_FAILED: session=%r action=%s conn=%r err=%s -- buffering",
                    session_id, getattr(payload, "action", "?"), getattr(conn, "id", "?"), e,
                )
                if self._connections.get(session_id) is conn:
                    self._connections.pop(session_id, None)
        else:
            logger.warning(
                "REPLY_BUFFERED: session=%r action=%s -- no live connection; "
                "will redeliver on reconnect.",
                session_id, getattr(payload, "action", "?"),
            )
        self._buffer_reply(session_id, payload)

    def _claim_slot(self, session_key: str, conn: ServerConnection, tab_id: str | None, *, sender: bool) -> bool:
        """Point the session's reply slot at ``conn``; return whether it did.

        A message always claims the slot: replies go to whoever asked. A
        heartbeat re-claims it only for the same tab (a reconnect after a
        dropped socket) or when no live connection holds it, never from a
        different tab that happens to share the browser's session key.
        """
        current = self._connections.get(session_key)
        owner_tab = self._slot_tab.get(session_key)
        if not sender and current is not None and current is not conn:
            if tab_id is not None and owner_tab is not None and tab_id != owner_tab:
                return False
        self._connections[session_key] = conn
        self._slot_tab[session_key] = tab_id
        return True

    def _buffer_reply(self, session_id: str, payload: Payload) -> None:
        """Buffer an undeliverable reply for redelivery, bounded per session."""
        with self._outbox_lock:
            queue = self._outbox.setdefault(session_id, [])
            queue.append(payload)
            overflow = len(queue) - self._outbox_max
            if overflow > 0:
                del queue[0:overflow]  # drop oldest

    def _flush_outbox(self, session_id: str, conn: ServerConnection) -> None:
        """Redeliver buffered replies to a freshly-(re)claimed live connection.

        Called at every slot reclaim (USER_MESSAGE / __heartbeat) so a reply
        buffered during a reconnect gap reaches the browser on its next beat.
        """
        with self._outbox_lock:
            pending = self._outbox.pop(session_id, None)
        if not pending:
            return
        logger.info(
            "REPLY_FLUSH: session=%r delivering %d buffered reply(ies) -> conn=%r",
            session_id, len(pending), getattr(conn, "id", "?"),
        )
        for i, payload in enumerate(pending):
            try:
                conn.send(json.dumps(payload, cls=PayloadEncoder))
            except Exception as e:
                # Still failing -- re-buffer this + the rest (order preserved)
                # so a later reclaim retries, and stop.
                logger.warning(
                    "REPLY_FLUSH_FAILED: session=%r err=%s -- re-buffering %d",
                    session_id, e, len(pending) - i,
                )
                with self._outbox_lock:
                    queue = self._outbox.setdefault(session_id, [])
                    queue[0:0] = pending[i:]
                return

    def reply(self, session: Session, message: str) -> None:
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message_obj: Message = Message(t=MessageType.STR, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_STR,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_markdown(self, session: Session, message: str) -> None:
        """Send an agent reply to a specific user, containing text in Markdown format.

        Args:
            session (Session): the user session
            message (str): the message in Markdown format to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message_obj: Message = Message(t=MessageType.MARKDOWN, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_MARKDOWN,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_html(self, session: Session, message: str) -> None:
        """Send an agent reply to a specific user, containing text in HTML format.

        Args:
            session (Session): the user session
            message (str): the message in HTML format to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message_obj: Message = Message(t=MessageType.HTML, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_HTML,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)
        
    def reply_file(self, session: Session, file: File) -> None:
        """Send a file reply to a specific user

        Args:
            session (Session): the user session
            file (File): the file to send
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message_obj: Message = Message(t=MessageType.FILE, content=file.get_json_string(), is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_FILE,
                          message=file.to_dict(),
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_image(self, session: Session, img: np.ndarray) -> None:
        """Send an image reply to a specific user.

        Before being sent, the image is encoded as jpg and then as a base64 string. This must be known before dedocing
        the image on the client side.

        Args:
            session (Session): the user session
            img (np.ndarray): the image to send
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        retval, buffer = cv2.imencode('.jpg', img)  # Encode as JPEG
        base64_img = base64.b64encode(buffer).decode('utf-8')
        message_obj: Message = Message(t=MessageType.IMAGE, content=base64_img, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_IMAGE,
                          message=base64_img,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_dataframe(self, session: Session, df: DataFrame) -> None:
        """Send a DataFrame agent reply, i.e. a table, to a specific user.

        Args:
            session (Session): the user session
            df (pandas.DataFrame): the message to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message = df.to_json() 
        #TODO processor will check for JSON instead of Dataframe, so the processor needs to convert to DF
        message_obj: Message = Message(t=MessageType.DATAFRAME, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_DF,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_options(self, session: Session, options: list[str]):
        """Send a list of options as a reply. They can be used to let the user choose one of them

        Args:
            session (Session): the user session
            options (list[str]): the list of options to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        d = {}
        for i, button in enumerate(options):
            d[i] = button
        #TODO processor should also process the individual strings in the list of strings
        message = json.dumps(d)
        message_obj: Message = Message(t=MessageType.OPTIONS, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_OPTIONS,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_plotly(self, session: Session, plot: plotly.graph_objs.Figure) -> None:
        """Send a Plotly figure as an agent reply, to a specific user.

        Args:
            session (Session): the user session
            plot (plotly.graph_objs.Figure): the message to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        message = plotly.io.to_json(plot)
        message_obj: Message = Message(t=MessageType.PLOTLY, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_PLOTLY,
                          message=message,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_location(self, session: Session, latitude: float, longitude: float) -> None:
        """Send a location reply to a specific user.

        Args:
            session (Session): the user session
            latitude (str): the latitude of the location
            longitude (str): the longitude of the location
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        location_dict = {'latitude': latitude, 'longitude': longitude}
        message_obj: Message = Message(t=MessageType.LOCATION, content=location_dict, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_LOCATION,
                          message=location_dict,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_task_list_update(self, session: Session, tasks: list) -> None:
        """Send a snapshot of the reasoning state's task list to a specific user.

        Emitted alongside (not in place of) the existing
        ``task_added`` / ``task_completed`` / ``task_skipped`` reasoning step
        events, so a client can render a live "task panel" without having to
        reconstruct the list from the step stream. The
        ``reasoning_started`` / ``reasoning_finished`` reasoning steps bracket
        the lifetime of this panel.

        Persisted in the dedicated ``reasoning_step`` table (kind=
        ``task_list_update``) — not in the chat table — so it does not
        pollute chat history retrieval.

        Args:
            session (Session): the user session.
            tasks (list): the current task list as a list of dicts with shape
                ``{"id": int, "description": str, "status": str, "result": str}``.
                Each call carries the *full* current snapshot — the client
                replaces its state rather than applying a diff.
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        timestamp = datetime.now()
        session.save_reasoning_event('task_list_update', tasks)
        payload = Payload(action=PayloadAction.AGENT_REPLY_TASK_LIST_UPDATE,
                          message=tasks,
                          timestamp=timestamp)
        self._send(session.id, payload)

    def reply_reasoning_step(self, session: Session, step: dict) -> None:
        """Send an intermediate reasoning-state step to a specific user.

        A *reasoning step* is any observable event the predefined reasoning
        state body produces while it is working: the LLM emitting tool calls
        or text, a tool returning a result, the task list being mutated
        (planned / completed / skipped), the orchestrator pushing back a
        premature final answer, or the step budget being exhausted. Sending
        these to the client lets the UI show a live "thinking" trace before
        the final reply arrives.

        Persisted in the dedicated ``reasoning_step`` table (kind=
        ``reasoning_step``) — not in the chat table — so it does not pollute
        chat history retrieval.

        Args:
            session (Session): the user session.
            step (dict): the step payload with the shape
                ``{"kind": str, "step": int, "summary": str, "details": dict}``.
                ``kind`` is one of ``llm_text``, ``llm_tool_calls``,
                ``tool_result``, ``task_added``, ``task_completed``,
                ``task_skipped``, ``pushback`` or ``max_steps``. ``summary``
                is a short human-readable description suitable for direct
                rendering; ``details`` is a kind-specific structured object
                for richer client-side rendering.
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        timestamp = datetime.now()
        session.save_reasoning_event('reasoning_step', step)
        payload = Payload(action=PayloadAction.AGENT_REPLY_REASONING_STEP,
                          message=step,
                          timestamp=timestamp)
        self._send(session.id, payload)

    def reply_rag(self, session: Session, rag_message: RAGMessage) -> None:
        """Send a rag reply to a specific user.

        Args:
            session (Session): the user session
            rag_message (RAGMessage): the rag message to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        rag_message_dict = rag_message.to_dict()
        message_obj: Message = Message(t=MessageType.RAG_ANSWER, content=rag_message_dict, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_RAG,
                          message=rag_message_dict,
                          timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_speech(self, session: Session, message: str, audio_speed: float = None) -> None:
        """Send an audio reply to a specific user.

        The text message is converted to speech and sent to the user. Before being sent, the audio is encoded as a
        Base64 string. This must be taken into account when decoding the audio on the client side.

        Args:
            session (Session): the user session
            message (str): the text message to be converted to speech and sent to the user.
            audio_speed (float, optional): The speed of the audio. If not provided, the speed is retrieved from the
            session, or defaults to 1.0. 0.5 is half speed, 2.0 is double speed, etc. Defaults to None.
        """
        
        audio_dict = session._agent.nlp_engine.text2speech(session, message)
        audio_array = audio_dict['audio']
        sample_rate = audio_dict['sampling_rate']
        dtype = audio_array.dtype
        shape = audio_array.shape

        # TODO: the sped up / slowed down audio sounds like in an echo chamber, not so good
        # Adjust audio speed if needed, preserving pitch (avoid chipmunk effect)
        if audio_speed and audio_speed != 1.0:
            # librosa expects float32 audio
            audio_array = audio_array.astype(np.float32)

            if audio_array.ndim == 1:
                # Mono audio
                audio_array = librosa.effects.time_stretch(audio_array, rate=audio_speed)

            elif audio_array.ndim == 2:
                # Multi-channel audio: librosa doesn't support stereo directly,
                # so apply time_stretch per channel and pad to equal lengths if needed
                channels = []
                for ch in range(audio_array.shape[0]):
                    stretched = librosa.effects.time_stretch(audio_array[ch], rate=audio_speed)
                    channels.append(stretched)
                
                # Find the length of the longest stretched channel
                max_len = max([c.shape[0] for c in channels])
                
                # Pad shorter channels with zeros to match
                channels = [np.pad(c, (0, max_len - c.shape[0])) for c in channels]
                
                audio_array = np.stack(channels, axis=0)

            # Update shape and dtype after speed change
            shape = audio_array.shape
            dtype = audio_array.dtype

        audio_array_contiguous = np.ascontiguousarray(audio_array)
        audio_bytes = audio_array_contiguous.tobytes()
        base64_bytes = base64.b64encode(audio_bytes)
        base64_string_audio = base64_bytes.decode('utf-8')

        message = {
            "audio_data_base64": base64_string_audio,
            "metadata": {
                "sample_rate": sample_rate,
                "dtype": str(dtype),
                "shape": shape
            }
        }

        message_obj: Message = Message(t=MessageType.AUDIO, content=message, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_AUDIO, message=message, timestamp=message_obj.timestamp)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_gui(self, session: Session, gui: 'AgentGUI') -> None:
        """Send a GUI model reply to a specific user.

        The GUI model is serialized to JSON before being sent. It can be used to dynamically
        render a user interface on the client side.

        If ``gui.persist`` is ``True``, an entry keyed by ``gui.id`` is created in the session's
        ``gui_inputs`` dictionary so that subsequent GUI events from this component will have their
        input values persisted automatically and remain accessible via ``session.gui_inputs``.

        Args:
            session (Session): the user session
            gui (AgentGUI): the GUI model to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        if gui.persist:
            session.gui_inputs[gui.id] = {}
        ui_json = gui_to_json(gui)
        message_obj: Message = Message(t=MessageType.GUI, content=ui_json, is_user=False, timestamp=datetime.now())
        session.save_message(message_obj)
        payload = Payload(action=PayloadAction.AGENT_REPLY_GUI,
                          message=ui_json,
                          timestamp=message_obj.timestamp,
                          message_id=gui.id)
        payload.message = self._agent.process(session=session, message=payload.message, is_user_message=False)
        self._send(session.id, payload)

    def reply_gui_update(self, session: Session, gui: 'AgentGUI') -> None:
        """Push an updated GUI model to the client.

        Used when the session's GUI model changes during interaction (e.g. via
        :meth:`~baf.core.session.Session.set_gui`). Clients should replace their current GUI
        state with the received model.

        Unlike :meth:`reply_gui`, this method does **not** persist the message to the chat history.

        Args:
            session (Session): the user session
            gui (AgentGUI): the updated GUI model to send to the user
        """
        if session.platform is not self:
            raise PlatformMismatchError(self, session)
        ui_json = gui_to_json(gui)
        payload = Payload(action=PayloadAction.AGENT_REPLY_GUI_UPDATE,
                          message=ui_json,
                          timestamp=datetime.now())
        self._send(session.id, payload)
