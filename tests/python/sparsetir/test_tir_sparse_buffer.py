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
import tvm
import tvm.tir as tir

def test_axis_creation():
    i = tir.sparse.DenseFixedAxis('i', 128)
    j = tir.sparse.DenseFixedAxis('j', 128)
    k = tir.sparse.DenseFixedAxis('k', 128)
    print(i, j, k)


if __name__ == "__main__":
    test_axis_creation()
