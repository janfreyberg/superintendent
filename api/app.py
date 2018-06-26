#!/usr/bin/env python

"""Flask app for labelling.

Requires a config file with the following contents.

[database]
; dialect+driver://username:password@host:port/database

dialect=xxx
driver=xxx
username=xxx
password=xxx
host=xxx
port=xxx
database=xxx

[server]
debug=true
port=8000

"""

import configparser
import os

from flask import Flask, jsonify
from werkzeug.exceptions import (HTTPException, InternalServerError,
                                 default_exceptions)

from superintendent.distributed.dbqueue import Backend

# Config


app = Flask(__name__)


def json_errorhandler(exception):
    """Create a JSON-encoded flask Response from an Exception."""

    if not isinstance(exception, HTTPException):
        exception = InternalServerError()

    response = jsonify({
        'error': exception.name,
        'description': exception.description,
        'code': exception.code
    })
    response.status_code = exception.code

    return response


for code in default_exceptions.keys():
    app.register_error_handler(code, json_errorhandler)


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.ini'
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

if config['server']['debug'].lower() == 'true':
    debug = True
else:
    debug = False
port = int(config['server']['port'])


def get_backend():
    return Backend.from_config_file(
        CONFIG_PATH, storage_type='integer_index'
    )


# Routes


@app.route('/example', methods=['GET'])
def example():
    """Get example to label."""
    q = get_backend()
    example_id, integer_index = q.pop()
    return jsonify(
        {'example_id': example_id, 'integer_index': integer_index}
    ), 200


@app.route('/label/example/<example_id>/value/<value>', methods=['POST'])
def label(example_id, value):
    """Queue a multiplication and return the task ID."""
    q = get_backend()
    q.submit(example_id, value)
    return jsonify({'example_id': example_id, 'value': value}), 200


# Main

if __name__ == '__main__':
    app.run(debug=debug, port=port)
