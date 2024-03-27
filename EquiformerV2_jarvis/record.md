/nfs/homedirs/xzh/equiformer_v2/oc20/trainer/energy_trainer_v2.py  
energy_trainer_v2  

self.normalizer = self.config["dataset"]  

self.use_auxiliary_task = False  

self.name="is2re"  

train method

eval_every: 5000     from config    

primary_metric: self.evaluator.task_primary_metric[self.name]  
这里self.name = 'is2re' -> primary_metric = 'energy_mae'  

self.step: 最初是0  

amp原来是True  

run_dir在sh脚本中自己设定  
在这个命令行脚本中，`--run-dir` 参数指定了一个用于存储运行期间产生的所有输出的目录。这包括训练模型的权重、日志文件、配置文件副本等。具体来说：

- `--run-dir 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@8x2'`: 这个参数设置 `run-dir`（运行目录）为一个特定的文件路径。在这种情况下，路径是 `'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@8x2'`。

这个路径中包含了许多信息，可能反映了用于此次运行的特定配置设置。例如，它可能包含了模型的名称、批大小（bs@64）、学习率（lr@2e-4）、权重衰减（wd@1e-3）、训练周期数（epochs@12）、预热周期（warmup-epochs@0.1）和GPU设置（g@8x2）。这些细节有助于快速识别和引用特定的训练运行和其结果。

总的来说，`run-dir` 在这个脚本中被用作一个组织和存储训练过程输出的地方，使得管理和访问这些输出变得更加方便和直观。   


