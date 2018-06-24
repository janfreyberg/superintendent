from flask import jsonify
from werkzeug.exceptions import HTTPException, InternalServerError


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
