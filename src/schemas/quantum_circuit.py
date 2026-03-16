"""Pydantic schemas for Quantum Circuit structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class QuantumOperationSpec(BaseModel):
    gate: str
    row: Optional[int] = None
    column: int = 0
    controlRow: Optional[int] = None
    targetRow: Optional[int] = None
    controlRow2: Optional[int] = None
    label: Optional[str] = None
    height: Optional[int] = None


class SingleQuantumGateSpec(BaseModel):
    """Schema for a single quantum gate operation."""
    operation: QuantumOperationSpec


class SystemQuantumCircuitSpec(BaseModel):
    """Schema for a complete quantum circuit system."""
    qubitCount: int = Field(default=2, ge=1)
    algorithmName: str = ""
    operations: List[QuantumOperationSpec] = Field(min_length=1)


# -- Modification schema --

class QuantumModificationSpec(BaseModel):
    """Schema for quantum circuit modification operations."""
    mode: Literal["append", "replace"] = "append"
    qubitCount: Optional[int] = Field(default=None, ge=1)
    operations: List[QuantumOperationSpec] = Field(min_length=1)
