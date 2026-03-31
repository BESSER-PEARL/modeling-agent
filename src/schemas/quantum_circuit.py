"""Pydantic schemas for Quantum Circuit structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class QuantumOperationSpec(BaseModel):
    gate: str = Field(
        description="Quantum gate name (e.g. H, X, Y, Z, CNOT, CZ, SWAP, TOFFOLI, MEASURE, BARRIER).",
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
        description="Zero-based qubit index of the control qubit (for controlled gates).",
    )
    targetRow: Optional[int] = Field(
        default=None,
        description="Zero-based qubit index of the target qubit (for controlled gates).",
    )
    controlRow2: Optional[int] = Field(
        default=None,
        description="Zero-based qubit index of the second control qubit (for TOFFOLI).",
    )
    label: Optional[str] = Field(
        default=None,
        description="Display label for FUNCTION/ORACLE/UNITARY gates.",
    )
    height: Optional[int] = Field(
        default=None,
        description="Number of qubits the gate spans vertically (for multi-qubit gates).",
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
        description="Name of the quantum algorithm.",
    )
    operations: List[QuantumOperationSpec] = Field(
        min_length=1,
        description="Ordered list of quantum gate operations in the circuit.",
    )


# -- Modification schema --

class QuantumModificationSpec(BaseModel):
    """Schema for quantum circuit modification operations."""
    mode: Literal["append", "replace"] = Field(
        default="append",
        description="Modification mode: append or replace.",
    )
    qubitCount: Optional[int] = Field(
        default=None,
        ge=1,
        description="New total number of qubits.",
    )
    operations: List[QuantumOperationSpec] = Field(
        min_length=1,
        description="Quantum gate operations to append or replace with.",
    )
