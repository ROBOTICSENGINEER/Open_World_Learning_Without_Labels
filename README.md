# Open-World Learning Without Labels
Open-World Learning Without Labels


Mohsen Jafarzadeh, Akshay Raj Dhamija, Chunchun Li, Steve Cruz, Touqeer Ahmad, Terrance E. Boult


Open-world learning is a problem where an autonomous agent detects things that it does not know and learns them over time from a non-stationary and never-ending stream of data; in an open-world environment, the training data and objective criteria are never available at once. The agent should grasp new knowledge from learning without forgetting acquired prior knowledge. Researchers proposed a few open-world learning agents for image classification tasks that operate in complex scenarios. However, all prior work on open-world learning has all labeled data to learn the new classes from the stream of images. In scenarios where autonomous agents should respond in near real-time or work in areas with limited communication infrastructure, human labeling of data is not possible. Therefore, supervised open-world learning agents are not scalable solutions for such applications. Herein, we propose a new framework that enables agents to learn new classes from a stream of unlabeled data in an unsupervised manner. Also, we study the robustness and learning speed of such agents with supervised and unsupervised feature representation. We also introduce a new metric for open-world learning without labels. We anticipate our theories and method to be a starting point for developing autonomous true open-world never-ending learning agents.


# How to run

### A) Training and saving feature (inside train folder)
1. Train supervised EfficientNet-B3 on ImageNet 2012  datasetusing `efficient_b3_fp16.py` or `efficient_b3_fp32.py`
2. Train self-supervised EfficientNet-B3 using MoCo V2 on ImageNet 2012 and Places2 Dataset:  `moco_Imagenet.py` and `moco_places.py`
3. Extract feautur from EfficientNet-B3 using `feature_saver_joint.py`
4. Train EVM using `train_evm_cosine.py`


### B) Run and evaluate open world learning 

1. run open-world unsupervised learning inside `OWL_without_label`
2. run open-world unsupervised learning inside `OWL_with_label`
3. evaluate the resulkt using scripts in `eval`


# Non overlapping classes


You can see the list of (166 classes) of Imagnet 2010 that are not non-overlapping with ImageNet 2012 classes in `data/new_166.txt`. Also you can use `data/new_166_dict.json`.



# License

Copyright (c) 2020 Mohsen Jafarzadeh. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by Mohsen Jafarzadeh.
4. Neither the name of the Mohsen Jafarzadeh nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY MOHSEN JAFARZADEH "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MOHSEN JAFARZADEH BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




