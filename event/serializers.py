from rest_framework import serializers


class UploadPhotoSerializer(serializers.Serializer):
    album_id = serializers.IntegerField()


class SearchByFaceSerializer(serializers.Serializer):
    image = serializers.ImageField()


class UploadedPhotoSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    filename = serializers.CharField()
    faces_found = serializers.IntegerField()


class UploadPhotoResponseSerializer(serializers.Serializer):
    uploaded = serializers.IntegerField()
    photos = UploadedPhotoSerializer(many=True)


class MatchedPhotoSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.CharField()
    album = serializers.CharField()


class SearchByFaceResponseSerializer(serializers.Serializer):
    matched_count = serializers.IntegerField()
    matched_photos = MatchedPhotoSerializer(many=True)
