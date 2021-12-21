# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""SparseTIR axes and SparseBuffer
"""
from typing import Dict, List, Optional

import tvm._ffi
from tvm.ir import PrimExpr
from tvm.runtime import Object
from tvm.tir import Var

from . import _ffi_api
from .buffer import Buffer


@tvm._ffi.register_object("tir.sparse.Axis")
class Axis(Object):
    """Base class of all the sparse axes."""

    @property
    def name(self):
        return _ffi_api.GetAxisName(self)

    @property
    def length(self):
        return _ffi_api.GetAxisLength(self)

    @property
    def idtype(self):
        return _ffi_api.GetAxisIndexType(self)
    
    @property
    def nnz(self):
        return _ffi_api.GetNNZ(self)


@tvm._ffi.register_object("tir.sparse.DenseAxis")
class DenseAxis(Axis):
    pass


@tvm._ffi.register_object("tir.sparse.SparseAxis")
class SparseAxis(Axis):
    pass


@tvm._ffi.register_object("tir.sparse.DenseFixedAxis")
class DenseFixedAxis(DenseAxis):
    """DenseFixedAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis
    """

    name: str
    length: PrimExpr

    def __init__(self, name, length):
        self.__init_handle_by_constructor__(_ffi_api.DenseFixedAxis, name, length)  # type: ignore


@tvm._ffi.register_object("tir.sparse.DenseFromSparseAxis")
class DenseFromSparseAxis(DenseFixedAxis):
    """DenseFromSparseAxis node

    Parameters
    ----------
    base : Axis
        The based sparse axis.
    """
    
    base: Axis

    def __init__(self, base):
        self.__init_handle_by_constructor__(_ffi_api.DenseFromSparseAxis, base)  # type: ignore


@tvm._ffi.register_object("tir.sparse.FusedAxis")
class FusedAxis(DenseFixedAxis):
    """FusedAxis node

    Parameters
    ----------
    group : List[Axis]
        The axes group to be fused.
    index : int
        The index of current axis in the fused axes group.
    """

    group: List[Axis]
    index: int

    def __init__(self, group, index):
        self.__init_handle_by_constructor__(_ffi_api.FusedAxis, group, index)  # type: ignore


@tvm._ffi.register_object("tir.sparse.DenseVariableAxis")
class DenseVariableAxis(DenseAxis):
    """DenseVariableAxis node

    Parameters
    ----------
    name : str
        The name of the axis
    
    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    indptr : Buffer
        The indptr buffer of the axis
    """

    name: str
    parent: Axis
    length: PrimExpr
    nnz: PrimExpr
    indptr: Buffer

    def __init__(self, name, parent, length, nnz, indptr):
        self.__init_handle_by_constructor__(
            _ffi_api.DenseVariableAxis, name, parent, length, nnz, indptr  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.AttachedAxis")
class AttachedAxis(DenseVariableAxis):
    """AttachedAxis node

    Parameters
    ----------
    name : str
        The name of the axis.
    parent : Axis
        The axis to attach to.
    orig : Axis
        The axis to be attached.
    nnz : PrimExpr
        The number of nonzeros of the returned axis.
    indptr : PrimExpr
        The new indptr array of the the returned axis.
    """

    name : str
    parent : Axis
    orig : Axis
    nnz : PrimExpr
    indptr : PrimExpr
    
    def __init__(self, name, parent, length, nnz, indptr):
        self.__init_handle_by_constructor__(
            _ffi_api.AttachedAxis, name, parent, length, nnz, indptr
        )


@tvm._ffi.register_object("tir.sparse.SparseFixedAxis")
class SparseFixedAxis(DenseAxis):
    """SparseFixedAxis node

    Parameters
    ----------
    name : str
        The name of the axis

    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    indices : Buffer
        The indices buffer of the axis

    nnz_cols : PrimExpr
        The fixed number of non-zero elements along the axis
    """

    name: str
    parent: Axis
    length: PrimExpr
    indices: Buffer
    nnz_cols: PrimExpr

    def __init__(self, name, parent, length, indices, nnz_cols):
        self.__init_handle_by_constructor__(
            _ffi_api.SparseFixedAxis, name, parent, length, indices, nnz_cols  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.SparseVariableAxis")
class SparseVariableAxis(DenseAxis):
    """SparseVariableAxis node

    Parameters
    ----------
    name : str
        The name of the axis
    
    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    indptr : Buffer
        The indptr buffer of the axis

    indices : Buffer
        The indices buffer of the axis
    """

    name: str
    parent: Axis
    length: PrimExpr
    indptr: Buffer
    indices: Buffer

    def __init__(self, name, parent, length, indptr, indices):
        self.__init_handle_by_constructor__(
            _ffi_api.SparseVariableAxis, name, parent, length, indptr, indices  # type: ignore
        )


@tvm._ffi.register_object("tir.sparse.SparseBuffer")
class SparseBuffer(Object):
    """SparseBuffer node

    Parameters
    ----------
    axes : List[Axis]
        The axes of the sparse buffer

    data : Buffer
        The data of the sparse buffer

    name : str
        The name of the sparse buffer
    """

    axes: List[Axis]
    data: Buffer
    name: str

    def __init__(self, axes, data, name):
        self.__init_handle_by_constructor__(_ffi_api.SparseBuffer, axes, data, name)  # type: ignore


@tvm._ffi.register_object("tir.sparse.SpIterVar")
class SpIterVar(Object):
    """IterVar in SparseTIR

    Parameters
    ----------
    var : Var
        The var of the SpIterVar

    max_extent : PrimExpr
        The maximum extent of the SpIterVar

    is_reduction : bool
        Whether the SpIterVar is a reduction iterator

    axis : Axis
        The axis over which the SpIterVar iterates
    """

    var: Var
    max_extent: PrimExpr
    is_reduction: bool
    axis: Axis

    DenseFixed = 0
    DenseVariable = 1
    SparseFixed = 2
    SparseVariable = 3

    def __init__(self, var, max_extent, is_reduction, axis):
        self.__init_handle_by_constructor__(
            _ffi_api.SpIterVar, var, max_extent, is_reduction, axis  # type: ignore
        )
