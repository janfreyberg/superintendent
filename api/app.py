import os

from flask import Flask, jsonify
from werkzeug.exceptions import (HTTPException, InternalServerError,
                                 default_exceptions)

from superintendent.distributed.dbqueue import Backend

app = Flask(__name__)


DATABASE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../tests/config.ini'
)


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


def get_backend():
    return Backend.from_config_file(
        DATABASE_CONFIG_PATH, storage_type='integer_index'
    )


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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
