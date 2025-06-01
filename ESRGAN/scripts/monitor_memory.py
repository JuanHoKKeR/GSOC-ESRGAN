import subprocess
import time
import psutil

def monitor_training():
    """Monitorea memoria durante entrenamiento"""
    while True:
        # GPU
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu', 
                '--format=csv,nounits,noheader'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                used, total, util = result.stdout.strip().split(', ')
                gpu_percent = int(used) / int(total) * 100
                print(f"🔥 GPU: {used}/{total}MB ({gpu_percent:.1f}%) | Util: {util}%")
                
                if gpu_percent > 95:
                    print("⚠️  ADVERTENCIA: GPU cerca del límite!")
                    
        except Exception as e:
            print(f"Error GPU: {e}")
        
        # RAM
        ram = psutil.virtual_memory()
        print(f"💾 RAM: {ram.used//1024//1024}/{ram.total//1024//1024}MB ({ram.percent:.1f}%)")
        
        if ram.percent > 90:
            print("⚠️  ADVERTENCIA: RAM cerca del límite!")
        
        print("-" * 50)
        time.sleep(10)

if __name__ == "__main__":
    monitor_training()