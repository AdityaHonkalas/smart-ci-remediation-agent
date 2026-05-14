#!/usr/bin/python3

# ------- Log collector for failed GHA workflows ---------- 

'''
 ## Log collection pipeline
 1. Point to the target repository
 2. REST API call for retreiving the list of workflows
 3. REST API call for retreiving the failed workflow runs from each of the workflow
 4. Python function for parsing the data from API response 
'''