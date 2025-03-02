#!/bin/bash

# Get sudo
#sudo -v

# Install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Enable brew
(
  echo
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"'
) >>/Users/$(whoami)/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Install packages and casks with brew
echo "Installing programs with homebrew"
brew update
brew upgrade

# Original applications
brew install --cask 1password 1password-cli orbstack raycast rectangle-pro shottr visual-studio-code

# ML-specific applications
brew install --cask anaconda docker jupyter-notebook-viewer google-chrome nvidia-cuda # Add CUDA support if you have NVIDIA hardware
brew install --cask slack 
brew install --cask obsidian 
brew install --cask microsoft-excel 
brew install --cask lens # Kubernetes GUI management tool

# Original command line tools
brew install corepack deno dockutil fnm gh git httpie iperf3 node plow stripe tfenv tmux fzf

# ML-specific command line tools
brew install python@3.11 python@3.10
brew install poetry pipenv pyenv 
brew install jupyterlab 
brew install r 
brew install cmake 
brew install awscli 
brew install postgresql 
brew install ffmpeg 
brew install htop glances 
brew install wget curl
brew install jq 
brew install graphviz 
brew install pandoc
brew install gcc gfortran # Compilers needed for some scientific packages

# C++ development tools
brew install gcc clang llvm cmake ninja boost boost-python eigen
brew install gdb lldb
brew install cppcheck clang-format
brew install catch2 googletest fmt nlohmann-json spdlog
brew install opencv 
brew install sfml # Graphics library, useful for visualizations
brew install valgrind # For memory leak detection
brew install ccache # Speed up compilation

# Rust development tools
brew install rustup-init # Rust installer and version management
rustup-init -y # Install Rust with defaults
source $HOME/.cargo/env # Add Rust to PATH for this session

# Install additional Rust tools
cargo install cargo-watch 
cargo install cargo-edit 
cargo install cargo-expand 
cargo install cargo-update 
cargo install cargo-outdated # Shows outdated dependencies
cargo install cargo-audit # Security vulnerabilities checking
cargo install cargo-bloat # Find what takes most space
cargo install cargo-tree # Dependency tree visualization
cargo install tokei # Statistics about your code
cargo install hyperfine # Benchmarking tool

# Install Rust Analysis for VSCode
rustup component add rust-analysis rust-src rustfmt clippy

# Kubernetes tools
brew install kubectl # Kubernetes command-line tool
brew install kubectx # Easily switch between clusters and namespaces
brew install kustomize # Kubernetes native configuration management
brew install helm # Kubernetes package manager
brew install k9s # Terminal UI for Kubernetes
brew install kubecolor # Colorized kubectl output
brew install kubeshark # API traffic viewer for Kubernetes
brew install argo # Kubernetes workflow engine for ML pipelines
brew install stern # Multi-pod log tailing for Kubernetes
brew install kind # Run local Kubernetes clusters using Docker
brew install minikube # Local Kubernetes environment

# Install Kubeflow CLI if needed for ML on Kubernetes
wget https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_darwin.tar.gz
tar -xvf kfctl_v1.2.0-0-gbc038f9_darwin.tar.gz
chmod +x kfctl
sudo mv kfctl /usr/local/bin

# Install ML frameworks via pip as they're often better managed this way
pip3 install numpy pandas scikit-learn matplotlib seaborn tensorflow torch torchvision transformers huggingface_hub keras opencv-python plotly xgboost lightgbm nltk spacy gensim statsmodels jupyter ray mlflow optuna wandb scipy

# Install Kubeflow related Python packages
pip3 install kfp kfserving seldon-core

# create LaunchAgents dir
mkdir -p ~/Library/LaunchAgents

# enable automatic updates every 12 hours
echo "Enabling autoupdate for homebrew packages..."
brew tap homebrew/autoupdate
brew autoupdate start 43200 --upgrade

# hidapitester -- used for controlling logitech litra lights
curl -L -o hidapitester-macos-arm64.zip https://github.com/todbot/hidapitester/releases/latest/download/hidapitester-macos-arm64.zip &&
  unzip hidapitester-macos-arm64.zip &&
  sudo mv hidapitester /usr/local/bin/

# Set up dock icons
echo "Setting up dock"
dockutil --remove all --no-restart
dockutil --add "/Applications/Visual Studio Code.app" --no-restart
dockutil --add "/Applications/Docker.app" --no-restart # Add Docker
dockutil --add "/Applications/Lens.app" --no-restart # Add Kubernetes Lens
dockutil --add "/Applications/Jupyter Notebook.app" --no-restart # Add Jupyter
dockutil --add "/System/Applications/Utilities/Terminal.app" --no-restart
dockutil --add "/Applications/Slack.app" --no-restart # Add Slack
dockutil --add "/System/Applications/Messages.app" --no-restart
dockutil --add "/System/Applications/Notes.app" --no-restart
dockutil --add "/Applications/Obsidian.app" --no-restart # Add Obsidian
dockutil --add "/System/Applications/Utilities/Activity Monitor.app" --no-restart
dockutil --add "/System/Applications/System Settings.app" --no-restart

# Folders to add to the dock
dockutil --add '/Applications' --view grid --display folder --no-restart
dockutil --add '~/Documents' --view list --display folder --no-restart
dockutil --add '~/Downloads' --view list --display folder

# xcode command line tools
xcode-select --install

# oh-my-tmux
cd ~
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .

eval "$(op signin)"

# git config
echo "Setting up git"

git config --global user.name "Arhum Zafar"
git config --global user.email "arhumzafar@yahoo.com"
git config --global core.editor "code --wait"
git config --global push.default upstream

# commit signing with 1password
git config --global user.signingkey "$(op item get "Github SSH Key" --fields "Public Key" --account )"
git config --global gpg.format "ssh"
#git config --global gpg.ssh.program "/Applications/1Password.app/Contents/MacOS/op-ssh-sign"
git config --global commit.gpgsign true

# git aliases
git config --global alias.undo "reset --soft HEAD^"

# set up ssh keys
echo "Setting up SSH keys"
mkdir -p ~/.ssh
op read "op://Private/Github SSH Key/private key" -o ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-add ~/.ssh/id_ed25519

# Set up dock hiding if on a laptop
dockconfig() {
  printf "\nLaptop selected, setting up dock hiding."
  defaults write com.apple.dock autohide -bool true
  defaults write com.apple.dock autohide-delay -float 0
  defaults write com.apple.dock autohide-time-modifier -float 0
  killall Dock
}

read -n1 -p "[D]esktop or [L]aptop? " systemtype
case $systemtype in
d | D) printf "\nDesktop selected, no special dock config." ;;
l | L) dockconfig ;;
*) echo INVALID OPTION, SKIPPING ;;
esac

# add karabiner mappings
echo "Getting karabiner configs"
mkdir -p ~/.config/karabiner/
curl -# https://gist.githubusercontent.com/markflorkowski/bc393361c0222f19ec3131b5686ed080/raw/62aec7067011cdf5e90cf54f252cbfb5a1e49de0/karabiner.json -o ~/.config/karabiner/karabiner.json
curl -# https://gist.githubusercontent.com/markflorkowski/3774bbbfeccd539c4343058e0740367c/raw/7c6e711a9516f83ff48c99e43eef9ca13fb05246/1643178345.json -o ~/.config/karabiner/assets/complex_modifications/1643178345.json

# configure rectangle pro to use icloud sync and launch on login
echo "Updating RectanglePro config"
/usr/libexec/PlistBuddy -c 'delete :iCloudSync' /Users/$(whoami)/Library/Preferences/com.knollsoft.Hookshot.plist
/usr/libexec/PlistBuddy -c 'add :iCloudSync bool true' /Users/$(whoami)/Library/Preferences/com.knollsoft.Hookshot.plist
/usr/libexec/PlistBuddy -c 'delete :launchOnLogin' /Users/$(whoami)/Library/Preferences/com.knollsoft.Hookshot.plist
/usr/libexec/PlistBuddy -c 'add :launchOnLogin bool true' /Users/$(whoami)/Library/Preferences/com.knollsoft.Hookshot.plist

echo "Updating macOS settings"

# Avoid the creation of .DS_Store files on network volumes or USB drives
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true

# Enable three-finger drag
defaults write com.apple.AppleMultitouchTrackpad DragLock -bool false
defaults write com.apple.AppleMultitouchTrackpad Dragging -bool false
defaults write com.apple.AppleMultitouchTrackpad TrackpadThreeFingerDrag -bool true

# Dock tweaks
defaults write com.apple.dock orientation -string left # Move dock to left side of screen
defaults write com.apple.dock show-recents -bool FALSE # Disable "Show recent applications in dock"
defaults write com.apple.Dock showhidden -bool TRUE    # Show hidden applications as translucent
killall Dock

# Finder tweaks
defaults write NSGlobalDomain AppleShowAllExtensions -bool true            # Show all filename extensions
defaults write com.apple.finder FXEnableExtensionChangeWarning -bool false # Disable warning when changing a file extension
defaults write com.apple.finder FXPreferredViewStyle Clmv                  # Use column view
defaults write com.apple.finder AppleShowAllFiles -bool true               # Show hidden files
defaults write com.apple.finder ShowPathbar -bool true                     # Show path bar
defaults write com.apple.finder ShowStatusBar -bool true                   # Show status bar
killall Finder

# Disable "the disk was not ejected properly" messages
defaults write /Library/Preferences/SystemConfiguration/com.apple.DiskArbitration.diskarbitrationd.plist DADisableEjectNotification -bool YES
killall diskarbitrationd

# Set up VS Code extensions for data science and ML
echo "Installing VS Code extensions for ML/Data Science, C++, and Rust"
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension njpwerner.autodocstring
code --install-extension donjayamanne.python-environment-manager
code --install-extension kevinrose.vsc-python-indent
code --install-extension mechatroner.rainbow-csv
code --install-extension grapecity.gc-excelviewer
code --install-extension bierner.markdown-preview-github-styles
code --install-extension yzhang.markdown-all-in-one
code --install-extension mhutchie.git-graph
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools # Kubernetes extension

# C++ extensions
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension xaver.clang-format
code --install-extension cschlosser.doxdocgen
code --install-extension matepek.vscode-catch2-test-adapter
code --install-extension akiramiyakoda.cppincludeguard
code --install-extension ms-vscode.makefile-tools

# Rust extensions
code --install-extension rust-lang.rust-analyzer
code --install-extension serayuzgur.crates
code --install-extension tamasfe.even-better-toml
code --install-extension vadimcn.vscode-lldb
code --install-extension swellaby.vscode-rust-test-adapter

# Create conda environments for different ML scenarios
echo "Setting up Conda environments for ML"
conda create -n tensorflow python=3.10 -y
conda create -n pytorch python=3.10 -y
conda create -n general-ml python=3.11 -y
conda create -n mlops python=3.11 -y # New environment for MLOps and Kubernetes
conda create -n cpp-ml python=3.10 -y # Environment for C++ ML integrations

# Set up the PyTorch environment
conda activate pytorch
pip install torch torchvision matplotlib pandas scikit-learn jupyter seaborn
conda deactivate

# Set up the general ML environment
conda activate general-ml
pip install numpy pandas scikit-learn matplotlib seaborn jupyter statsmodels nltk spacy gensim xgboost lightgbm mlflow optuna wandb plotly
conda deactivate

# Set up the MLOps environment with Kubernetes tools
conda activate mlops
pip install kfp kfserving seldon-core kubeflow-fairing kubernetes mlflow ray[tune] bentoml docker prefect
conda deactivate

# Set up C++ ML environment
conda activate cpp-ml
conda install -c conda-forge pybind11 cxx-compiler
pip install cmake scikit-build
pip install torch tensorflow tensorflow_hub
conda deactivate

# Set up Kubernetes configs directory
mkdir -p ~/.kube

# Create project directories
mkdir -p ~/MLProjects
mkdir -p ~/KubernetesConfigs
mkdir -p ~/CppProjects
mkdir -p ~/RustProjects

# Create basic C++ ML project template
mkdir -p ~/CppProjects/template/{src,include,lib,test,build}

# Create CMakeLists.txt for C++ template
cat > ~/CppProjects/template/CMakeLists.txt << EOL
cmake_minimum_required(VERSION 3.14)
project(MyProject VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build tests" ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# Main library
add_library(MyProject
    src/main.cpp
)

target_include_directories(MyProject PUBLIC
    \${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(MyProject PUBLIC
    Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# Main executable
add_executable(MyProjectApp src/main.cpp)
target_link_libraries(MyProjectApp PRIVATE MyProject)

# Tests
if(BUILD_TESTS)
    find_package(Catch2 REQUIRED)
    enable_testing()
    add_executable(tests test/test_main.cpp)
    target_link_libraries(tests PRIVATE MyProject Catch2::Catch2)
    include(CTest)
    include(Catch)
    catch_discover_tests(tests)
endif()
EOL

# Create example C++ file
cat > ~/CppProjects/template/src/main.cpp << EOL
#include <iostream>
#include <Eigen/Dense>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // Example using Eigen for matrix operations
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    
    // Using fmt for formatting
    fmt::print("Hello, {}!\n", "Machine Learning Engineer");
    
    // Using spdlog for logging
    spdlog::info("Matrix:\n{}", m);
    
    // Using JSON for data
    json j = {
        {"name", "ML Model"},
        {"type", "regression"},
        {"parameters", {
            {"learning_rate", 0.01},
            {"max_depth", 10}
        }}
    };
    
    std::cout << "Model config: " << j.dump(4) << std::endl;
    
    return 0;
}
EOL

# Create example Rust project
cd ~/RustProjects
cargo new ml_utils
cd ml_utils

# Add dependencies to Cargo.toml
cat > Cargo.toml << EOL
[package]
name = "ml_utils"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
ndarray-stats = "0.5"
ndarray-rand = "0.14"
nalgebra = "0.32"
plotters = "0.3"
rand = "0.8"
rayon = "1.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
clap = { version = "4.3", features = ["derive"] }
EOL

# Create example Rust ML utility
cat > src/main.rs << EOL
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use serde::{Serialize, Deserialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize)]
struct MLModel {
    name: String,
    features: Vec<String>,
    weights: Vec<f64>,
    bias: f64,
}

impl MLModel {
    fn predict(&self, features: &Array1<f64>) -> f64 {
        let weights = Array1::from_vec(self.weights.clone());
        weights.dot(features) + self.bias
    }
    
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }
}

fn generate_sample_data(n_samples: usize, n_features: usize, noise: f64) -> (Array2<f64>, Array1<f64>) {
    let x = Array::random((n_samples, n_features), Uniform::new(0., 10.));
    
    // Create true weights and bias
    let true_weights = Array1::random(n_features, Uniform::new(-1.0, 1.0));
    let true_bias = 0.5;
    
    // Generate target values with some noise
    let noise_term = Array::random(n_samples, Uniform::new(-noise, noise));
    let y = x.dot(&true_weights) + true_bias + noise_term;
    
    (x, y)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Generate sample data
    let (x, y) = generate_sample_data(100, 3, 0.5);
    
    // Print some stats
    println!("X shape: {:?}", x.shape());
    println!("Y shape: {:?}", y.shape());
    println!("X max value: {:?}", x.max()?);
    println!("Y max value: {:?}", y.max()?);
    
    // Create a simple model
    let model = MLModel {
        name: "SimpleLinear".to_string(),
        features: vec!["feature1".to_string(), "feature2".to_string(), "feature3".to_string()],
        weights: vec![0.5, -0.2, 0.8],
        bias: 0.1,
    };
    
    // Make a prediction
    let sample = Array1::from_vec(vec![2.0, 3.0, 1.0]);
    let prediction = model.predict(&sample);
    println!("Prediction for {:?}: {}", sample, prediction);
    
    // Save the model
    model.save("simple_model.json")?;
    println!("Model saved to simple_model.json");
    
    Ok(())
}
EOL

# Start Minikube for local testing
minikube start

echo "Starting services"
open "/Applications/Rectangle Pro.app"
open "/Applications/Karabiner-Elements.app"
open "/Applications/Shottr.app"
open "/Applications/Docker.app" # Start Docker
open "/Applications/Lens.app" # Start Kubernetes Lens

echo "Removing config programs"
brew remove dockutil

# oh-my-zsh (must be last)
sh -c "$(curl -# -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# zsh aliases
echo "alias c='open \$1 -a \"Visual Studio Code\"'" >>~/.zshrc
echo "alias jlab='jupyter lab'" >>~/.zshrc
echo "alias py='python3'" >>~/.zshrc
echo "alias nb='jupyter notebook'" >>~/.zshrc
echo "alias tf='conda activate tensorflow'" >>~/.zshrc
echo "alias pt='conda activate pytorch'" >>~/.zshrc
echo "alias ml='conda activate general-ml'" >>~/.zshrc
echo "alias mlops='conda activate mlops'" >>~/.zshrc
echo "alias cppml='conda activate cpp-ml'" >>~/.zshrc
echo "alias dockerclean='docker system prune -a'" >>~/.zshrc
echo "alias k='kubectl'" >>~/.zshrc
echo "alias kn='kubens'" >>~/.zshrc
echo "alias kx='kubectx'" >>~/.zshrc
echo "alias mk='minikube'" >>~/.zshrc
echo "alias mkd='minikube dashboard'" >>~/.zshrc
echo "alias klogs='stern'" >>~/.zshrc

# C++ and Rust aliases
echo "alias cmakebuild='mkdir -p build && cd build && cmake .. && make'" >>~/.zshrc
echo "alias cmakerelease='mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make'" >>~/.zshrc
echo "alias cmakerebuild='rm -rf build && mkdir -p build && cd build && cmake .. && make'" >>~/.zshrc
echo "alias cmaketest='cd build && ctest'" >>~/.zshrc
echo "alias crun='cargo run'" >>~/.zshrc
echo "alias cbuild='cargo build'" >>~/.zshrc
echo "alias ctest='cargo test'" >>~/.zshrc
echo "alias crelease='cargo build --release'" >>~/.zshrc
echo "alias ccheck='cargo check'" >>~/.zshrc
echo "alias cfmt='cargo fmt'" >>~/.zshrc
echo "alias cclippy='cargo clippy'" >>~/.zshrc
echo "alias cupdate='cargo update'" >>~/.zshrc

# add ssh-agent, conda, kubectl and Rust plugins to zsh
sed -i -e 's/plugins=(git)/plugins=(git ssh-agent conda-auto-env kubectl helm kube-ps1 rust cargo)/' ~/.zshrc

# Enable kube-ps1 prompt
echo 'PROMPT=$PROMPT"\$(kube_ps1) "' >>~/.zshrc
echo "kubeoff" >>~/.zshrc # Disabled by default, can enable with kubeon

# Add Rust to PATH permanently
echo 'source $HOME/.cargo/env' >>~/.zshrc

# fnm stuff
echo "eval \"\$(fnm env --use-on-cd)\"" >>~/.zshrc

# fzf
source <(fzf --zsh)

# Setup auto-activation of conda environments
mkdir -p ~/.oh-my-zsh/custom/plugins/conda-auto-env
curl -# https://raw.githubusercontent.com/esc/conda-zsh-completion/master/conda-auto-env.plugin.zsh -o ~/.oh-my-zsh/custom/plugins/conda-auto-env/conda-auto-env.plugin.zsh

# Add conda to path
echo "# >>> conda initialize >>>" >>~/.zshrc
echo "export PATH=/opt/homebrew/anaconda3/bin:\$PATH" >>~/.zshrc
echo "# <<< conda initialize <<<" >>~/.zshrc

# Create a basic .condarc file
echo "auto_activate_base: false" > ~/.condarc
echo "channels:" >> ~/.condarc
echo "  - conda-forge" >> ~/.condarc
echo "  - defaults" >> ~/.condarc

# Create a basic Kubernetes deployment template for ML models
cat > ~/KubernetesConfigs/ml-deployment-template.yaml << EOL
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model-container
        image: your-ml-model-image:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
          requests:
            cpu: "500m"
            memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
EOL

# finish
source ~/.zshrc
