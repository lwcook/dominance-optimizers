#!/bin/bash

nosetests --cover-package=domopt --with-coverage --cover-erase

coverage-badge -o tests/coverage.svg -f

coverage report -m
