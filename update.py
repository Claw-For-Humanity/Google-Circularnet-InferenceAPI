import os
import subprocess

def git_pull():
    repo_dir = './Google-Circularnet-Server-Integration'
    # Change directory to your repository
    os.chdir(repo_dir)
    # Run the git pull command
    subprocess.run(['git', 'pull'])

if __name__ == "__main__":
    git_pull()
