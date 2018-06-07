import os
from argparse import ArgumentParser
from nipype import Function, Node, Workflow, IdentityInterface
# TODO: 
#     1. collect bayes factor scores and either put it into one file to feed to R or
#         collect it within R and have it as an R object
#     2. run the brain to diagnosis step with the bayes factor as prior logodds
#     3. save results
