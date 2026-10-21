import torch, time, statistics as st, json
torch.backends.cudnn.benchmark=False
H=384
dev="cuda"
def gpu_mhz():
    try: return int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read())//10**6
    except: return -1
lstm=torch.nn.LSTM(H,H,batch_first=True).half().to(dev).eval()
def bench(N,T,reps=7):
    x=torch.randn(N,T,H,device=dev,dtype=torch.half)
    with torch.inference_mode():
        for _ in range(2): lstm(x)          # warm
        torch.cuda.synchronize()
        ts=[]
        for _ in range(reps):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); lstm(x); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b)/1e3)
    return st.median(ts), min(ts), max(ts)
# ramp the DVFS clock
x=torch.randn(128,2048,H,device=dev,dtype=torch.half)
t0=time.time()
with torch.inference_mode():
    while time.time()-t0<8: lstm(x)
torch.cuda.synchronize()
print("gpu MHz after warmup:",gpu_mhz())
out=[]
print("\n== batch scaling, T=2048 (one LSTM layer, fp16, cuDNN)")
print(f"{'N':>5} {'time_ms':>9} {'us/step':>9} {'TFLOP/s':>8} {'ms/chunk':>9} {'MHz':>5}")
for N in (8,16,32,64,96,128,160,192):
    t,lo,hi=bench(N,2048)
    fl=2*N*2048*(4*H)*(H+H)
    print(f"{N:>5} {t*1e3:>9.1f} {t/2048*1e6:>9.1f} {fl/t/1e12:>8.2f} {t/N*1e3:>9.3f} {gpu_mhz():>5}")
    out.append(dict(kind="N",N=N,T=2048,s=t,lo=lo,hi=hi,mhz=gpu_mhz()))
print("\n== sequence-length scaling, N=128")
print(f"{'T':>5} {'time_ms':>9} {'us/step':>9} {'TFLOP/s':>8}")
for T in (256,512,1024,2048,4096):
    t,lo,hi=bench(128,T)
    fl=2*128*T*(4*H)*(H+H)
    print(f"{T:>5} {t*1e3:>9.1f} {t/T*1e6:>9.1f} {fl/t/1e12:>8.2f}")
    out.append(dict(kind="T",N=128,T=T,s=t,lo=lo,hi=hi,mhz=gpu_mhz()))
json.dump(out,open("lstm_scaling.json","w"),indent=1)
