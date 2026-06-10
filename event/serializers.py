from rest_framework import serializers


class UploadPhotoSerializer(serializers.Serializer):
    album_id = serializers.IntegerField()
    image = serializers.ImageField()


class SearchByFaceSerializer(serializers.Serializer):
    image = serializers.ImageField()
    event_id = serializers.IntegerField(required=False, allow_null=True)


class UploadPhotoResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    filename = serializers.CharField()
    faces_found = serializers.IntegerField()


class MatchedPhotoSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.CharField()
    album = serializers.CharField()


class SearchByFaceResponseSerializer(serializers.Serializer):
    matched_count = serializers.IntegerField()
    matched_photos = MatchedPhotoSerializer(many=True)
