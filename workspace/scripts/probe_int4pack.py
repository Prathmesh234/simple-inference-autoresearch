import torch, time
dev="cuda"; M,N,K=128,14336,4096
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
# uint8 packed weight [N, K//2], 2 nibbles/byte
wpk = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
for ikt in (2, 4, 8):
    try:
        wp = torch._convert_weight_to_int4pack(wpk, ikt)
        ok_shape = tuple(wp.shape)
    except Exception as e:
        print(f"  innerKTiles={ikt}: convert ERR {str(e)[:100]}"); continue
    for gs in (32, 64, 128):
        nG = K // gs
        szz = torch.randn(nG, N, 2, device=dev, dtype=torch.bfloat16)
        try:
            def run(): return torch._weight_int4pack_mm(x, wp, gs, szz)
            y = run(); torch.cuda.synchronize()
            for _ in range(20): run()
            torch.cuda.synchronize(); t=time.perf_counter()
            for _ in range(300): run()
            torch.cuda.synchronize()
            print(f"  ikt={ikt} group={gs}: {(time.perf_counter()-t)/300*1e6:.1f} us  (int8 W8A8 ~57us)")
        except Exception as e:
            print(f"  ikt={ikt} group={gs}: mm ERR {str(e)[:100]}")
