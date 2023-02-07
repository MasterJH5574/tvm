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
# pylint: disable=redefined-builtin, invalid-name
"""Relax binary arithmetic and comparison operators."""
from . import _ffi_api
from ..expr import Expr

###################### Arithmetic operators ######################


def add(x1: Expr, x2: Expr) -> Expr:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.

    Examples
    --------
    .. code:: python

      bb = relax.BlockBuilder()
      a = relax.Var("a", relax.TensorStructInfo(shape=(2, 3), dtype="float32"))
      b = relax.Var("b", relax.TensorStructInfo(shape=(2, 1), dtype="float32"))
      c = bb.normalize(relax.op.add(a, b))  # c has TensorStructInfo(shape=(2, 3), dtype="float32")
    """
    return _ffi_api.add(x1, x2)  # type: ignore


def multiply(x1: Expr, x2: Expr) -> Expr:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.multiply(x1, x2)  # type: ignore
