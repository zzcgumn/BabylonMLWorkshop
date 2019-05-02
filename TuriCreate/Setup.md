
Apple's official documentation can be found here, https://github.com/apple/turicreate. They give a number of entry level examples of to fit models using Turi Create.

Messing around with the macOS inbuilt Python environment can create some really big headaches. You probably want work with another installation. Turi Create supports Python 3.6 and 2.7, you can check which version you are using with `python -V`. It is recommended to set up virtual environment for each python project. This makes it easier to avoid conflicts between installed packages.

Installing Turi Create in a python 2.7 virtual environment.
1. Create a workspace directory, e.g. `cd ~/; mkdir CoreMLWorkshop; cd CoreMLWorkshop`
2. Install virtualenv if you have not already got it `pip install virtualenv`
3. Create a virtual environment `virtualenv venv`
4. Activate the virtual environment `source  venv/bin/activate`
5. Install Turi Create in your environment `pip install -U turicreate`

Brew will by default install Python 3.7. To have both Python 3.7 and Python 3.6 installed (See this Stack Overflow answer, https://stackoverflow.com/questions/51125013/how-can-i-install-a-previous-version-of-python-3-in-macos-using-homebrew)

```
brew install python3
brew unlink python
brew install --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
brew link --overwrite python
```

Create a python 3 virtual environment
1. Create a workspace directory, e.g. `cd ~/; mkdir CoreMLWorkshop; cd CoreMLWorkshop`
2. Create a virtual environment `python3.6 -m venv venv`
3. Activate the virtual environment `source  venv/bin/activate`
4. Install Turi Create in your environment `pip install -U turicreate`

Use the command `deactivate` to leave a virtual environment.
