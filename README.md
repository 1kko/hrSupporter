# HR Supporter

HR Supporter uses local on-device LLM to analyze and generate interviewee's
- Timeline of job and scholar archievenemts
- Overall Evaluation (Technical Skills, Experience, Education, Projects, Achievements)
- Interview questions
by reading PDF resume, and output to markdown.

# Requirements
- Python 3.12+
- Poetry

# System Environment
- Nvidia Graphic cards with 12GB+ RAM
- WSL2 (used to test and run, but probably run other OS)

# How to install
Because we will process all private information, environment to run local(on-device) LLM is required.

## 1. Install Ollama
Visit https://ollama.com/download/ and select your OS, and install ollama.

After install, type following to pull `gemma2:9b` model.

```sh
$ ollama pull gemma2:9b
```

## 2. Install Poetry
If you already have installed package/dependency manager for poetry, you may skip this part.

Visit https://python-poetry.org/docs/ and install poetry with your preferred way.

```sh
$ pip install pipx
$ pipx install poetry
```


## 3. Clone this repository
Clone this repository and create 3 directories(`source`, `process`, `processed`) inside.

```sh
user@host:~$ git clone git://... hrSupporter
user@host:~$ cd hrSupporter
user@host:~/hrSupporter$ mkdir -p source process processed
```

## 4. Install dependencies.
Enter poetry shell mode, and install dependencies.
```sh
user@host:~/hrSupporter$ poetry shell
hrsupporter-py3.12 user@host:~/hrSupporter$ poetry install
```

## 5. Put your resume files into `source`. Once it's processed it will move PDF to `processed` directory.
```sh
hrsupporter-py3.12 user@host:~/hrSupporter$ cp ~/Downloads/*.pdf ./source
```

## 6. Run `process.py` to analyze the pdf files.
```sh
hrsupporter-py3.12 user@host:~/hrSupporter$ python ./process.py
```

# FAQ

## How to modify LLM prompts?
Open `process.py` with your favorite python editor, and locate `self.prompt_timeline = PromptTemplate(...`.

This is where all prompts are saved to hand over to LLM.

## How to change LLM?
Default model is `gemma2:9b`, however you may change to fit your taste.

Just locate `def main()` and change the model name. Make sure it's available by ollama.