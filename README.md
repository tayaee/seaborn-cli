# Explore Seaborn Example Plots via CLI

This project provides a command-line interface (CLI) for experimenting with example plots from the official [Seaborn documentation](https://seaborn.pydata.org/examples/index.html).  
It allows you to generate visualizations using Seaborn's built-in datasets — no coding required.

---

## Features

- Run Seaborn example plots directly from the command line
- Use built-in datasets like `iris`, `penguins`, `tips`, and more
- Save plots to image files with a single command
- Great for quick experimentation and learning Seaborn without writing Python code

---

## Getting Started

### 1. Install Python 3.10 or higher

```
python -V
```

### 2. Create a virtual environment and install dependencies
```
make-venv-uv.bat
```

### 3. Try Out Example Plots
```
sns --help                          # View available plot types and options
sns pairplot --help                # View options for the 'pairplot' command
sns pairplot --data=iris           # Generate a pairplot using the iris dataset
sns pairplot --data=iris --output=output.png  # Save the plot to an image file
```

## Supported Plot Types

```
sns --help
```
