# "TSQP: Safeguarding Real-Time Inference for Quantization Neural Networks on Edge Devices"

The code for implementing the ***TSQP: Safeguarding Real-Time Inference for Quantization Neural Networks on Edge Devices***.

## Requirements

### **Intel SGX Hardware** and **Gramine**
A device equipped with Intel SGX is required as the hardware to run the code. It is recommended to test on Linux since we have not tested Gramine in Windows.

The following steps are necessary to build a Gramine environment.

1. Linux-SGX Driver. SGX-Driver is required to be installed, which is the fundemental environment. Please refer to [Linux-SGX Respository](https://github.com/intel/linux-sgx) to build from source-code. For some versions of CPUs and systems, SGX may already be integrated in the system driver.

2. Gramine-SGX. Gramine-SGX is a libOS which supports runing application in SGX without modification. Please follow the [Gramine Respository](https://github.com/gramineproject/gramine) to install the Gramine.

3. Test. You can test your Gramine according to this simple [Demo](https://github.com/gramineproject/examples/tree/master/pytorch).

### Python Environment
Our code is tested in python 3.9, and is theoretically suitable for python >= 3.8. We provide a simple instruction to configure the essential python libraries.

```bash 
pip install -r requirements.txt
```


### Dataset
You are supposed to download [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet 2012](https://image-net.org/) or any dataset you want, and provide its location through the command line.

## Understanding TSQP
A easy case is shown in case_study.py.

## Running Experiments

### Inference Accuracy
Here, we have provided our correct dataset root or other settings in the ```/parameter_de_similarity/config.py```. Please provide your correct location in here or through the command line.
1. PTQ causes accuracy decrease on MobileNetv2. (about 45 min)

    One simple instruction:
    ```bash
    cd parameter_de_similarity/PTQ/bash && ./mobilenetv2.sh
    ```

    Try other model? We provide a ```all.sh``` (45 min per model):

    ```bash
    cd parameter_de_similarity/PTQ/bash && ./all.sh
    ```


2. Proposed Parameter De-Similarity

    One simple instruction (Over 30 hours):
    ```bash
    cd /parameter_de_similarity/reduce_range_adaptive_training && ./qat_bash.sh
    ```

    Single training only:
    ```bash
    cd /parameter_de_similarity/reduce_range_adaptive_training && python qat.py --model resnet18 --lr 1e-4 --epoches 30
    ```
     Other setting please refer to the code or ```config.py```.

### Inference Latency

Two terminals are required, one for REE and one for TEE. Maybe you need to recompile the grpc and configure the ```pytorch.manifest.template``` according to your own Gramine setting.

For our proposed TSQP, run the following commands (you can set the batch_size easily in the code):

1. VGG19
    ```bash
    Terminal 1 (REE)
    cd inference_latency/ours/vgg19
    python server.py
    ```

    ```bash
    Terminal 2 (TEE)
    cd inference_latency/ours/vgg19
    make clean && make SGX=1
    gramine-sgx ./pytorch client.py
    ```
2. ResNet18
    ```bash
    Terminal 1 (REE)
    cd inference_latency/ours/resnet18
    python server.py
    ```

    ```bash
    Terminal 2 (TEE)
    cd inference_latency/ours/resnet18
    make clean && make SGX=1
    gramine-sgx ./pytorch client.py
    ```
3. MobileNetv2
    ```bash
    Terminal 1 (REE)
    cd inference_latency/ours/mobilenetv2
    python server.py
    ```

    ```bash
    Terminal 2 (TEE)
    cd inference_latency/ours/mobilenetv2
    make clean && make SGX=1
    gramine-sgx ./pytorch client.py
    ```

For the state-of-the-art methods, the project structure is the same, for example:
```bash
Terminal 1 (REE)
cd inference_latency/aegisdnn/vgg19
python server.py
```

```bash
Terminal 2 (TEE)
cd inference_latency/aegisdnn/vgg19
make clean && make SGX=1
gramine-sgx ./pytorch client.py
```

### Inference Integrity

One simple instruction (about 10 min per model, 30 min in total):
```bash
cd inference_integrity && ./all.sh
```

Test on single model:
```bash
cd inference_integrity/{Model} && ./all.sh
```

### Model Confidentiality
Please refer to the [Official Code](https://github.com/tribhuvanesh/knockoffnets).

## Note

This is not the full and secure version, and it should not be used in any real production environments. To support real edge device, we are working on a full version with OP-TEE. It's coming soon...

## Citation
```
@inproceedings{sun2024tsqp,
  title={TSQP: Safeguarding Real-Time Inference for Quantization Neural Networks on Edge Devices},
  author={Sun, Yu and Xiong, Gaojian and Liu, Jianhua and Liu, Zheng and Cui, Jian},
  booktitle={2025 IEEE Symposium on Security and Privacy (SP)},
  pages={1--1},
  year={2024},
  organization={IEEE Computer Society}
}
```

## License
This project is released under the MIT License.
