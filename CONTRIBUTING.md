# Contributing to Heart Disease Prediction Project

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- **Clear title** describing the bug
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (Python version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:
- **Clear and descriptive title**
- **Detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **Code examples** if applicable

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** thoroughly
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

## Code Style

- Follow **PEP 8** style guide for Python code
- Use **meaningful variable names**
- Add **comments** for complex logic
- Include **docstrings** for functions
- Keep **functions focused** and single-purpose

## Areas for Contribution

Here are some ideas for contributions:

### High Priority
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Implement cross-validation
- [ ] Add model performance visualizations
- [ ] Create unit tests

### Medium Priority
- [ ] Deep learning model (Neural Network)
- [ ] Feature engineering experiments
- [ ] Model explainability (SHAP values)
- [ ] Interactive dashboard

### Nice to Have
- [ ] Web application (Flask/Streamlit)
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

## Testing

Before submitting a pull request:
- [ ] Run all notebook cells successfully
- [ ] Verify no errors in output
- [ ] Test with different random seeds
- [ ] Check documentation is up to date

## Commit Message Guidelines

Use clear and meaningful commit messages:

```
feat: add cross-validation for model evaluation
fix: correct data preprocessing issue with categorical features
docs: update README with installation instructions
refactor: improve code structure in preprocessing section
```

**Prefixes:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `style`: Code style changes (formatting)

## Questions?

Feel free to open an issue with your question or reach out via:
- GitHub Issues
- Email: [your-email@example.com]

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## Recognition

Contributors will be recognized in the project README. Thank you for helping make this project better! 🙏

---

**Happy Contributing!** 🚀
