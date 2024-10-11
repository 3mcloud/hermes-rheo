from setuptools import setup

from setuptools import setup

# Function to read requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="hermes_rheo",
    version="1.0.0",
    description="The hermes package is designed to complement and extend the piblin library, offering a set of transforms "
                "tailored for advanced rheological analysis within the piblin 'Dataset' framework.",
    author="Alessandro Perego - 3M",
    license="MIT",
    install_requires=parse_requirements('requirements.txt'),
)