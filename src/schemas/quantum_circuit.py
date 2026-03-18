"""Pydantic schemas for Quantum Circuit structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class QuantumOperationSpec(BaseModel):
    gate: str = Field(
        description=(
            "Quantum gate name. Valid values: H (Hadamard), X (Pauli-X / NOT), "
            "Y (Pauli-Y), Z (Pauli-Z), S (Phase), T (Pi/8), "
            "CNOT (Controlled-NOT), CZ (Controlled-Z), CY (Controlled-Y), "
            "SWAP, TOFFOLI (CCX), RX, RY, RZ, "
            "MEASURE, BARRIER, FUNCTION, ORACLE, UNITARY, "
            "INTERLEAVE, DEINTERLEAVE"
        ),
    )
    row: Optional[int] = Field(
        default=None,
        description="Zero-based qubit index for single-qubit gates",
    )
    column: int = Field(
        default=0,
        description="Zero-based time-step (column) position in the circuit",
    )
    controlRow: Optional[int] = Field(
        default=None,
        description="For controlled gates (CNOT, CZ, CY, TOFFOLI) only. Zero-based qubit index of the control qubit",
    )
    targetRow: Optional[int] = Field(
        default=None,
        description="For controlled gates (CNOT, CZ, CY, TOFFOLI) only. Zero-based qubit index of the target qubit",
    )
    controlRow2: Optional[int] = Field(
        default=None,
        description="For double-controlled gates (TOFFOLI) only. Zero-based qubit index of the second control qubit",
    )
    label: Optional[str] = Field(
        default=None,
        description="For FUNCTION/ORACLE/UNITARY gates. Display label describing the gate operation",
    )
    height: Optional[int] = Field(
        default=None,
        description="For FUNCTION/ORACLE/INTERLEAVE/DEINTERLEAVE gates. Number of qubits the gate spans vertically",
    )


class SingleQuantumGateSpec(BaseModel):
    """Schema for a single quantum gate operation."""
    operation: QuantumOperationSpec = Field(
        description="The quantum gate operation to perform",
    )


class SystemQuantumCircuitSpec(BaseModel):
    """Schema for a complete quantum circuit system."""
    qubitCount: int = Field(
        default=2,
        ge=1,
        description="Total number of qubits in the circuit (minimum 1)",
    )
    algorithmName: str = Field(
        default="",
        description="Name of the quantum algorithm (e.g. 'Grover', 'Shor', 'Bell State')",
    )
    operations: List[QuantumOperationSpec] = Field(
        min_length=1,
        description="Ordered list of quantum gate operations that make up the circuit",
    )


# -- Modification schema --

class QuantumModificationSpec(BaseModel):
    """Schema for quantum circuit modification operations."""
    mode: Literal["append", "replace"] = Field(
        default="append",
        description="Modification mode: 'append' adds operations to the existing circuit, 'replace' replaces the entire circuit",
    )
    qubitCount: Optional[int] = Field(
        default=None,
        ge=1,
        description="New total number of qubits. Only set when changing the qubit count",
    )
    operations: List[QuantumOperationSpec] = Field(
        min_length=1,
        description="List of quantum gate operations to append or replace with",
    )
