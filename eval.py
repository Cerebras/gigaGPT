# Copyright 2023 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from data import get_dataloader
from model import GPTModel

from cerebras.modelzoo.common.utils.run.cli_pytorch import get_params_from_args

def main():
    params = get_params_from_args()
    
    from cerebras.modelzoo.common.run_utils import main
    main(params, GPTModel, None, get_dataloader)

if __name__ == '__main__':
    main()

