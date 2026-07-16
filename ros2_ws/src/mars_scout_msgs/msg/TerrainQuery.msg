# Natural-language query issued by the operator.
# Stamped so we can correlate queries with bag recordings.

std_msgs/Header header

# The raw NL string, e.g. "find the rock that looks like a skull"
string query_text

# Unique ID so downstream nodes can match responses to queries
string query_id

# How many seconds the agent should keep searching before giving up (0 = unlimited)
float32 timeout_sec
