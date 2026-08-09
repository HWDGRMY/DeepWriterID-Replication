from setuptools import setup, find_packages

setup(
    name='deepwriterid-replication',
    version='1.0.0',
    # 自动找到 src/ 目录下的所有包
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # 项目的核心依赖库（与你的 environment.yml 一致）
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'opencv-python>=4.5.0',
        'tqdm>=4.62.0',
        'pyyaml>=5.4.1',
        'matplotlib>=3.3.0',
        'signatory>=1.2.6',
    ],
    author='HWDGRMY',
    author_email='2201454718@qq.com',
    description='DeepWriterID Replication on CASIA-OLHWDB (1019 Writers)',
    python_requires='>=3.8',
)