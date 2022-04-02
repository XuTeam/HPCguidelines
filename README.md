# 1 Code and data preparation
## 1.1 Github for code
你需要把代码上传到Github，并将我（xlliu2017）加入到库中（具有pull和push权限）。这样我可以把你的代码同步到KAUST HPC。同步完成后，你的代码在HPC的路径为 /home/liux0t/your repository/
## 1.2 Data 
如果代码需要特定数据，而且数据太大无法上传到github
### 1.2.1 公开数据集
我已经安装torchvision, 相关数据集参考
https://pytorch.org/vision/stable/datasets.html
### 1.2.2 自定义数据集
自行写一个 data download 程序并包含在你的code里面
### 1.2.3 
如果是非常大的数据集，可以申请IT部门下载，你需要提供数据的specs以供IT下载


# 2 Enviroment and software
## 2.1 Available Modules on HPC
一些module使用前需要申请，比如MATLAB。大部分软件只需要slurm文件里输入命令， 比如 module load cmake/3.16.1/gcc4.8.5 。关于slurm  更多命令使用请上网搜索



------------------------------------------------ 以下是所以软件 ------------------------------------------------
alphafold/2.0/singularity                            matlab/R2019a
alphafold/2.1.1/python3                              matlab/R2020a(default)
alphafold/2.1.1/python3_jupyter                      matlab/R2021a
amber/18/gnu6.4.0-cuda9.0.176                        meshlab/1.3.2
amber/18/ompi300-cuda90                              mitsuba/0.6.0
bonito/0.3.2                                         mumax/3.10
bwa/0.7.17/gnu-6.4.0                                 namd/2.13/cuda10-verbs-smp-icc17
ctffind/4.1.10                                       nanoporetech/workflow_python3.8
decimate/0.9.5                                       octopus/11.1/openmpi4.0.3-intel2020-cuda11.0
decimate/0.9.6(default)                              opencv/4.1.0/gnu-6.4.0
decimate/latest                                      parabricks/3.0.0.2
deepfri/1.0.0                                        parabricks/3.0.0.2_ext_lic
elasticsearch_ksl/2019                               parallelware/1.1.0
ffmpeg/4.1.3/gnu-6.4.0                               parallelware/1.2.0
funcy/1.12/conda3env                                 parallelware/1.3.0
funcy/1.12/python3.6.2env                            parallelware/1.4.0
gaussian16/b.01/precompiled                          paraview/5.6.0-openmpi3.0.0
gautomatch/0.56                                      paraview/5.8.1-openmpi3.0.0
gctf/1.06                                            peet/1.13.0
gnuplot/5.0.0                                        quantumespresso/6.4/pgi-17.10_cuda-9.0
gromacs/2018/openmpi-2.1.1-intel-2016-cuda           quantumespresso/6.5/pgi-17.10_cuda-9.0
gromacs/2018.8/intelmpi-2017-cuda9.2                 quantumespresso/6.6/pgi-20.1_cuda-10.2
gromacs/2018.8/openmpi-2.1.1-intel-2016-cuda         quantumespresso/6.7/pgi-20.1_cuda-10.2
gromacs/2020.4/icc20_cuda11                          relion/3.0/openmpi-3.1.2-gnu6.4.0_cuda9.2.148.1
gromacs/2020.4/openmpi4.0.3-icc20-cuda11             relion/3.0/openmpi4.0.1_gnu6.4.0_cuda10.1.105
gstreamer/1.12.3/gnu-6.4.0                           relion/3.0.8/openmpi4.0.1_gnu6.4.0_cuda10.1.105.test
gstreamer/1.16.0/gnu-6.4.0                           relion/3.0_beta/openmpi-3.1.2-gnu6.4.0_cuda9.2.148.1
guppy/3.1.5                                          relion/3.1.0/cuda11.0
guppy/3.3.0                                          relion/3.1_beta/openmpi-3.1.2-gnu6.4.0_cuda9.2.148.1
guppy/3.3.3                                          relion/4.0_beta/cuda11.0
guppy/3.4.1                                          relion/4.0_beta2/openmpi4.0.3_cuda10.1.243
guppy/6.0.1                                          resmap/1.95
imod/4.10.22                                         scipion/1.2.1
imod/4.11.12                                         scipion/2.0
juicer/1.5                                           spliceai/1.3.1
ktf/0.6(default)                                     spot-1d/1.0
ktf/0.7                                              torch/0.1.12
machine_learning/2021.09                             torch/0.4.0
machine_learning/2021.10(default)                    valgrind/3.15.0/gnu-6.4.0
madagascar-gpu/3.0.1/gnu6.4.0_cuda10.1               vesta/3.4.6
matlab/R2016b                                        vesta/3.5.7
matlab/R2017b                                        xcrysden/1.5.60
matlab/R2018a

------------------------------------------------- /sw/csgv/modulefiles/compilers --------------------------------------------------
cmake/3.16.1/gcc4.8.5                  java/8u162                             openmpi/4.0.3-pgi20.1-cuda10.2
cmake/3.17.2/intel-2019                java/9.0.1                             perl/5.20.1
cmake/3.18.4/gnu-6.4.0                 julia/1.3.0                            perl/5.22.4_gnu-640
cmake/3.19.2/gnu-6.4.0                 julia/1.5.2                            perl/5.26.1/gnu-6.4.0
cmake/3.22.2/gcc4.8.5                  llvm/9.0.0                             perl/5.26.1/intel-2017
cuda/10.0.130                          mpich/3.3/gnu6.4.0-cuda9.2             perl/5.26.3/multi-threaded
cuda/10.1.105                          nvhpc/20.7                             perl/5.30.0(default)
cuda/10.1.243                          nvhpc-byo-compiler/20.7                pgi/17.10
cuda/10.2.89                           nvhpc-nompi/20.7                       pgi/17.10_openmpi
cuda/11.0.1                            openmpi/2.1.1/intel-2016-cuda9.2.148.1 pgi/18.10
cuda/11.1.1                            openmpi/3.0.0/gnu-6.4.0                pgi/20.1
cuda/11.2.2(default)                   openmpi/3.0.0/pgi-17.10                python/2.7.10
cuda/9.0.176                           openmpi/3.1.2/gnu6.4.0_cuda10.1.243    python/2.7.14
cuda/9.2.148.1                         openmpi/3.1.2/gnu6.4.0_cuda9.2.148.1   python/2.7.14-intel2017
gcc/10.2.0                             openmpi/3.1.2/intel2018_cuda9.2.148.1  python/3.6.2
gcc/11.1.0                             openmpi/3.1.2/intel2019_cuda10.1.243   python/3.7.0(default)
gcc/6.4.0(default)                     openmpi/3.1.2-cuda10.1                 python/3.8.1
gcc/8.2.0                              openmpi/3.1.2-cuda10.2                 R/3.3.3/intel-2017
go/1.15.2                              openmpi/3.1.2-cuda11.0                 R/3.4.2/gnu-6.4.0
intel/2016                             openmpi/4.0.1/gnu6.4.0_cuda10.1.105    R/3.4.2/intel-2017
intel/2017                             openmpi/4.0.1/gnu6.4.0_cuda9.2.148.1   R/3.5.0/gnu-6.4.0
intel/2018                             openmpi/4.0.1/intel2017_cuda10.1.243   R/3.5.0/intel-2018
intel/2019                             openmpi/4.0.1/pgi-18.10                R/3.6.0/gnu-6.4.0
intel/2020(default)                    openmpi/4.0.3-cuda10.1                 R/3.6.0/intel-2019
intelmpi/2019                          openmpi/4.0.3-cuda10.2                 R/4.0.2/gnu-6.4.0
intelmpi/2020                          openmpi/4.0.3-cuda11.0                 R/4.0.4
java/11.0.6                            openmpi/4.0.3-cuda11.2.2(default)      R/4.1.1/gnu-6.4.0
java/8u131                             openmpi/4.0.3-intel2020-cuda11.0

---------------------------------------------------- /sw/csgv/modulefiles/libs ----------------------------------------------------
boost/1.65.1/openmpi-2.1.1-gcc-6.4.0      hdf5/1.10.1/openmpi2.1.1-intel2016        nccl/2.4.8
boost/1.65.1/openmpi-3.1.2-cuda11.0       hdf5/1.10.3/openmpi3.0.0-gnu6.4.0         nccl/2.4.8-cuda10.1
cairo/1.14.12/gnu-6.4.0                   hdf5/1.10.4                               scons/4.0.1/python-3.7
cairo/1.16.0/gnu-6.4.0                    lapack/3.8.0/gnu-6.4.0                    swig/4.0.1/gnu-6.4.0
cudnn/7.2.1(default)                      lapack/3.9.0-gnu6.4.0                     ucx/1.5.1/cuda10.1.243
cudnn/7.2.1-cuda9.2.148.1                 libxc/3.0.0/gnu-6.4.0                     ucx/1.8.0/cuda10.1.243
cudnn/7.5.0                               libxc/3.0.0/intel-2016                    ucx/1.8.0/cuda10.2.89
cudnn/7.5.0-cuda10.1.105                  libxc/4.0.3/gnu-6.4.0                     ucx/1.8.0/cuda11.0.1
cudnn/8.1.1-cuda11.2.2                    libxc/4.0.3/intel-2016                    ucx/1.9.0/cuda10.2.89
dl/2020                                   libxc/4.0.3/intel-2017                    ucx/1.9.0/cuda11.2.2(default)
eigen/3.3.7                               libxc/4.2.3/intel-2019                    zlib/1.2.11/gnu-6.4.0
fftw/3.3.7/openmpi4.0.3-cuda10.2-gcc6.4.0 likwid/5.2.0/gnu6.4.0-cuda11.2.2          zlib/1.2.11/intel-2017
gsl/2.4/gnu-6.4.0                         nccl/2.2.13(default)
hdf5/1.10.1/openmpi2.1.1-gnu6.4.0         nccl/2.2.13-cuda9.2.148.1

---------------------------------------------------- /sw/services/modulefiles -----------------------------------------------------
default-appstack         intelstack-optimized     oneapi-v2022.1           singularity/3.1          singularity/3.5(default)
gpustack                 legacy-DS                otp/0.0.1                singularity/3.2          singularity/3.6
gpustack-legacy          mpifileutils/0.11        reframe/3.6.0            singularity/3.3
intelstack-legacy        oneapi-v2021.4           singularity/3.0          singularity/3.4

## 2.2 Conda
你可以使用Conda 配置自己的环境，现有我配置的环境anaconda3, 包括 pytorch, torchvision, cuda等。尽量配置自己的环境， 参考
https://www.hpc.kaust.edu.sa/ibex/best-practices-using-conda-pip-ibex

# 3 Slurm
所有可执行命令和程序都需要通过Slurm系统提交，你需要写一个 yours.sh 脚本来执行你的环境命令和运行程序。以下是一个例子

#!/bin/bash
## SLURM Resource requirement:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=myjob
#SBATCH --output=myjob.%J.out
#SBATCH --error=myjob.%J.err
#SBATCH --time=8:00:00
#SBATCH --partition=batch
##指定GPU数量
#SBATCH --gres=gpu:a100:1 
## Required software list:
module load intel/2020
## Run the application:
echo "This job ran on $SLURM_NODELIST dated `date`";
./my_exe
##### 现有资源248 a100，一般任务使用#SBATCH --partition=batch就可以




batch*        up 14-00:00:0      2  down$ gpu203-23-l,gpu203-23-r
batch*        up 14-00:00:0      1  maint gpu108-02-l
batch*        up 14-00:00:0      2  down* cn113-34-l,cn113-35-l
batch*        up 14-00:00:0     27    mix gpu101-02-l,gpu101-02-r,gpu101-09-l,gpu101-09-r,gpu101-16-l,gpu101-16-r,gpu108-02-r,gpu108-09-l,gpu108-09-r,gpu108-16-l,gpu108-16-r,gpu108-23-l,gpu108-23-r,gpu109-02-l,gpu109-02-r,gpu109-09-l,gpu109-09-r,gpu109-16-l,gpu109-16-r,gpu109-23-l,gpu109-23-r,gpu201-02-l,gpu201-02-r,gpu201-09-l,gpu201-09-r,gpu201-16-l,gpu201-16-r
batch*        up 14-00:00:0     16   idle gpu201-23-l,gpu201-23-r,gpu202-02-l,gpu202-02-r,gpu202-09-l,gpu202-09-r,gpu202-16-l,gpu202-16-r,gpu202-23-l,gpu202-23-r,gpu203-02-l,gpu203-02-r,gpu203-09-l,gpu203-09-r,gpu203-16-l,gpu203-16-r
gpu24         up 1-00:00:00      1    mix gpu101-09-r
gpu_wide      up 14-00:00:0      2    mix gpu110-[02,09]
gpu_wide24    up 1-00:00:00      4  drain gpu102-[02,09],gpu110-[16,23]
gpu_wide24    up 1-00:00:00      3    mix gpu102-16,gpu110-[02,09]
gpu_wide24    up 1-00:00:00      1   idle gpu102-23
gpu4          up    4:00:00      2  down$ gpu203-23-l,gpu203-23-r
gpu4          up    4:00:00      1  maint gpu108-02-l
gpu4          up    4:00:00      4  drain gpu102-[02,09],gpu110-[16,23]
gpu4          up    4:00:00     29    mix gpu101-02-r,gpu101-09-l,gpu101-09-r,gpu101-16-l,gpu101-16-r,gpu102-16,gpu108-02-r,gpu108-09-l,gpu108-09-r,gpu108-16-l,gpu108-16-r,gpu108-23-l,gpu108-23-r,gpu109-02-l,gpu109-02-r,gpu109-09-l,gpu109-09-r,gpu109-16-l,gpu109-16-r,gpu109-23-l,gpu109-23-r,gpu110-[02,09],gpu201-02-l,gpu201-02-r,gpu201-09-l,gpu201-09-r,gpu201-16-l,gpu201-16-r
gpu4          up    4:00:00     16   idle gpu201-23-l,gpu201-23-r,gpu202-02-l,gpu202-02-r,gpu202-09-l,gpu202-09-r,gpu202-16-l,gpu202-16-r,gpu202-23-l,gpu202-23-r,gpu203-02-l,gpu203-02-r,gpu203-09-l,gpu203-09-r,gpu203-16-l,gpu203-16-r