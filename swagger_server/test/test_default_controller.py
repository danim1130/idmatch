# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.match_result import MatchResult  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_last_face_picture_get(self):
        """Test case for last_face_picture_get

        
        """
        response = self.client.open(
            '/api/lastFacePicture',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_last_id_picture_get(self):
        """Test case for last_id_picture_get

        
        """
        response = self.client.open(
            '/api/lastIdPicture',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_read_id_post(self):
        """Test case for read_id_post

        Extract information from the submitted card
        """
        data = dict(image=(BytesIO(b'some file data'), 'file.txt'))
        response = self.client.open(
            '/api/readId',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
