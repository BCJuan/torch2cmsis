from setuptools import setup

setup(
    name='torch2cmsis',
    packages=['torch2cmsis'],
    license='Apache 2.0',
    version='0.1',
    description='PyTorch to CMSIS-NN converter',
    author='Juan Borrego Carazo',
    author_email='bcjuan@protonmail.com',
    url='https://github.com/BCJuan/torch2cmsis',
    download_url='https://github.com/BCJuan/torch2cmsis/archive/v_0.1.tar.gz',
    keywords=['Pytorch', 'CMSSIS-NN', 'embedded', 'neural networks'],
    install_requires=[
        'numpy',
        'tqdm',
        'torch',
        'torchvision'],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    zip_safe=False,
    python_requires='>=3.6'
    )