##############################################################################

#File Name (CommentingTemplate.py)
#Authors (put your name here if you make any major revisions)

'''
Description (what does this node do?)

'''
'''
How do you call this node?
rosrun <PackageName> <NodeName (usually the name of the file)> <parameters>

'''

#Topics this node is subscribed to
#Topics this node publishes to
#Services this node uses
#Other dependencies?

##############################################################################

#include
...
...

#CONSTANTS (organize these as necessary)
#names for constants should be in ALL CAPS
...
...

##############################################################################

#Setup
#every node should have one
def Setup(params):
    '''
    body of setup function
    Run all prequisite code to prepare for the main loop.
    MAKE SURE YOUR EDITOR USES 4 SPACES FOR TABS
    '''

#Loop
#every node should have one
def Loop(params):
    '''
    body of the main loop
    MAKE SURE YOUR EDITOR USES 4 SPACES FOR TABS
    '''

##############################################################################

#Helper Functions

'''
function header
what does it return? what parameters? general description.
'''
def Foo(params):
    '''
    body of function
    MAKE SURE YOUR EDITOR USES 4 SPACES FOR TABS
    '''

##############################################################################


## Notes ## (not a part of the template)
# Other conventions

# All names should be descriptive of whatever it is!

# All function names should be camelcase and begin with a capital letter (FunctionName)
# All names of classes should be camelcase and begin with a capital letter (ClassName)
# All variable names should be camelcase and begin with a lowercase letter (variableName)
# All names for constants should be in all caps with underscores for spacing (CONSTANT_NAME)

#If you installed any libraries or other packages that your code depends on, make sure you make note of that!
