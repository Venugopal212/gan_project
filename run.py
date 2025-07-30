import os
import subprocess

def run_command(command, cwd=None):
    print(f"\n[Running] {command}")
    try:
        subprocess.run(command, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed: {command}")
        exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- PART 1: 2D GANs ---
    part1_dir = os.path.join(base_dir, "part1_2d_gans")
    print("\n===== PART 1: 2D GANs on Synthetic Data =====")
    
    run_command("python train_sine.py", cwd=part1_dir)
    run_command("python train_spiral.py", cwd=part1_dir)

    # --- PART 2: PathMNIST GANs ---
    part2_dir = os.path.join(base_dir, "part2_pathmnist")
    print("\n===== PART 2: PathMNIST GANs =====")

    run_command("python train_cgan.py", cwd=part2_dir)
    run_command("python train_dcgan.py", cwd=part2_dir)
    run_command("python visualize.py --type cgan --ckpt logs/cgan/ --out cgan_outputs/", cwd=part2_dir)
    run_command("python visualize.py --type dcgan --ckpt logs/dcgan/ --out dcgan_output.png", cwd=part2_dir)

    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    main()
