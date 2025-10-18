**Enabling low-cost RLHF using single AMD MI300x:**
```sh
docker run -e LOCAL_SIZE=1 -it --rm --hostname metro-rlhf-test --shm-size=8g \
      --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/kfd --device=/dev/dri --group-add=video \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /:/host -w /host$(pwd) \
      --entrypoint=/bin/bash tutelgroup/metro-rlhf-mi300x

root@metro-rlhf-test:$  git clone https://github.com/MetroRLHF/MetroRLHF && cd MetroRLHF
root@metro-rlhf-test:$  hf download Qwen/Qwen3-8B --local-dir Qwen/Qwen3-8B

root@metro-rlhf-test:$  DTYPE=float32 T=512   python3 ./MetroRLHF-Correctness-Validation.py  # Param=32GB (Total GMEM = 45GB) w MetroRLHF
root@metro-rlhf-test:$  DTYPE=float32 T=4096  python3 ./MetroRLHF-Correctness-Validation.py  # Param=32GB (Total GMEM = 104GB) w MetroRLHF
root@metro-rlhf-test:$  DTYPE=float32 T=0     python3 ./MetroRLHF-Correctness-Validation.py  # Param=32GB (Total GMEM = 165GB) w/o MetroRLHF

root@metro-rlhf-test:$  DTYPE=bfloat16 T=512  python3 ./MetroRLHF-Correctness-Validation.py  # Param=16GB (Total GMEM = 22GB) w MetroRLHF
root@metro-rlhf-test:$  DTYPE=bfloat16 T=4096 python3 ./MetroRLHF-Correctness-Validation.py  # Param=16GB (Total GMEM = 49GB) w MetroRLHF
root@metro-rlhf-test:$  DTYPE=bfloat16 T=0    python3 ./MetroRLHF-Correctness-Validation.py  # Param=16GB (Total GMEM = 78GB) w/o MetroRLHF

root@metro-rlhf-test:$  DTYPE=bfloat16 T=0    python3 ./Metro-Example-DPO.py  # Param=16GB (Total GMEM = 51GB) w/o MetroRLHF
root@metro-rlhf-test:$  DTYPE=bfloat16 T=512  python3 ./Metro-Example-DPO.py  # Param=16GB (Total GMEM = 22GB) w MetroRLHF
```

To use Inference-optimized MetroRLHF, please visit: https://github.com/microsoft/Tutel
