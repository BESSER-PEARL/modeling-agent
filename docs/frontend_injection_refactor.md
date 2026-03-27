# Frontend Injection Refactor Plan

## Current Architecture (Complex)

```
Backend sends systemSpec
    |
    v
Frontend handleInjection()
    |
    v
Wait for editor to mount (waitForModelingService)
    |
    v
UMLModelingService.processSystemSpec()
    |-- ConverterFactory.convertCompleteSystem()   <-- static, no editor needed
    |-- Calculate layout shift                     <-- needs current model (available in Redux)
    |-- Push undo snapshot                         <-- needs undo stack (standalone)
    |-- Return ModelUpdate
    |
    v
UMLModelingService.injectToEditor()
    |-- editor.model = newModel                    <-- direct editor write
    |
    v
Editor re-renders
```

**Problems:**
- Every injection blocks on `waitForModelingService()` (3s timeout)
- Modeling service depends on the Apollon editor being mounted
- Tab switching / new tab breaks because the editor ref is stale
- The converter is static but trapped inside the modeling service
- `activeModel` is sent separately from `projectSnapshot` (same data, sent twice)
- The editor is treated as the owner of the model instead of a renderer

## Target Architecture (Simple)

```
Backend sends systemSpec
    |
    v
ConverterFactory.convertCompleteSystem(systemSpec)   <-- static import, no editor
    |
    v
dispatch(updateDiagramModelThunk({ model }))         <-- write to Redux
    |
    v
dispatch(bumpEditorRevision())                       <-- editor re-creates with new model
    |
    v
Editor picks up model from Redux on mount
```

**No editor dependency. No timing issues. No stale refs.**

## What to Change

### 1. `useAssistantLogic.ts` — `handleInjection()`

Replace the modeling service path with converter + Redux for all actions.

**Before:**
```ts
// Wait for editor...
if (targetIsUml && !modelingServiceRef.current) {
    await waitForModelingService();   // blocks up to 3s
}

// Use modeling service to convert + inject
if (targetIsUml && modelingServiceRef.current) {
    switch (command.action) {
        case 'inject_complete_system':
            update = modelingServiceRef.current.processSystemSpec(...)
            break;
        case 'inject_element':
            update = modelingServiceRef.current.processSimpleClassSpec(...)
            break;
        case 'modify_model':
            update = modelingServiceRef.current.processModelModifications(...)
            break;
    }
    await modelingServiceRef.current.injectToEditor(update);
}
```

**After:**
```ts
import { ConverterFactory } from '../services/converters';

if (targetIsUml) {
    // Push undo snapshot before changing the model
    const currentModel = activeDiagram?.model;
    if (currentModel) {
        pushUndoSnapshot(currentModel, `Before ${command.action}`);
    }

    let newModel: any = null;

    switch (command.action) {
        case 'inject_complete_system':
            if (command.systemSpec) {
                const converter = ConverterFactory.getConverter(targetDiagramType);
                newModel = converter.convertCompleteSystem(command.systemSpec);
            }
            break;

        case 'inject_element':
            if (command.element) {
                const converter = ConverterFactory.getConverter(targetDiagramType);
                newModel = converter.convertSingleElement(command.element);
                // TODO: merge with existing model instead of replacing
            }
            break;

        case 'modify_model':
            // Modifications need the ModifierFactory (also static)
            // This handles add_attribute, rename, remove, etc.
            if (command.modifications || command.modification) {
                const modifier = ModifierFactory.getModifier(targetDiagramType);
                newModel = modifier.applyModifications(
                    currentModel,
                    command.modifications || [command.modification]
                );
            }
            break;
    }

    if (newModel) {
        await dispatch(updateDiagramModelThunk({ model: newModel }));
        dispatch(bumpEditorRevision());
        applied = true;
    }
}
```

### 2. `inject_element` — Merge Logic

For single element injection, the converter returns a model with just that element.
It needs to be **merged** with the existing model, not replace it.

Options:
- A: Converter returns a partial model, merge utility combines with existing
- B: Modifier handles "add element" as a modification to existing model
- C: Keep `processSimpleClassSpec` as a standalone utility (extract from modeling service)

Option C is simplest — the function just needs the current model and the converter, not the editor.

### 3. `modify_model` — Modifier Logic

The `ModifierFactory` is already mostly static. The modifiers read the current model,
apply changes (rename attribute, add class, remove relationship), and return the updated model.

Extract from modeling service into a standalone utility.

### 4. Stop Sending `activeModel` Separately

The `projectSnapshot` already contains the same model. Remove `activeModel` from the context payload.

**Frontend** (`AssistantClient.ts` / `useAssistantLogic.ts`):
- Remove `activeModel` from `buildWorkspaceContext()`
- Remove `compactContextPayload` model hashing logic
- Remove the `model` parameter from `sendMessage()` (it currently sends the model twice)

**Backend** (`protocol/adapters.py`):
- Stop reading `activeModel` from the payload
- `resolve_target_model()` already falls back to `projectSnapshot` — it will just always use that path

### 5. What Happens to `UMLModelingService`

After this refactor, the modeling service is only needed for:
- **Undo management** (`pushUndoSnapshot`, `popUndo`) — can be extracted to a standalone module
- **`getCurrentModel()`** — replaced by reading from Redux

It can be removed entirely or kept as a thin wrapper around the undo stack.

### 6. What Happens to `waitForModelingService()`

Deleted. No longer needed. The converter is static, Redux is always available.

## Files to Modify

| File | Change |
|------|--------|
| `useAssistantLogic.ts` | Rewrite `handleInjection()` to use converter + Redux |
| `useAssistantLogic.ts` | Remove `waitForModelingService()` |
| `useAssistantLogic.ts` | Remove `activeModel` from `buildWorkspaceContext()` |
| `AssistantClient.ts` | Remove `model` parameter from `sendMessage()` |
| `AssistantClient.ts` | Remove model hashing in `compactContextPayload()` |
| `UMLModelingService.ts` | Keep only undo logic, or remove entirely |
| `services/modifiers/` | Ensure modifiers work standalone (no editor dependency) |

## Files NOT Modified

| File | Why |
|------|-----|
| Backend (`execution.py`, handlers, etc.) | Backend already sends the right payload |
| `ConverterFactory` / converters | Already static, no changes needed |
| `workspaceSlice.ts` | Already has `updateDiagramModelThunk` + `bumpEditorRevision` |

## Migration Strategy

1. **Phase 1**: Use converter + Redux for `inject_complete_system` (same as current new-tab path but for all injections)
2. **Phase 2**: Extract modifier logic for `modify_model` and `inject_element`
3. **Phase 3**: Remove `activeModel` from frontend payload
4. **Phase 4**: Remove `UMLModelingService` or reduce to undo-only
