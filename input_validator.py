import pandas as pd

class ValidationError(Exception):
    """Base class for validation exceptions"""
    pass

class NullValueError(ValidationError):
    """Raised when value is null/NA"""
    pass

class ValueRangeError(ValidationError):
    """Raised when value is outside acceptable range"""
    pass

class InvalidTypeError(ValidationError):
    """Raised when value is of wrong type"""
    pass

class InvalidCategoryError(ValidationError):
    """Raised when value is not in allowed categories"""
    pass

class InputValidator:
    @staticmethod
    def validate_species(value):
        valid_species = {'Adelie', 'Gentoo', 'Chinstrap'}
        if pd.isna(value):
            raise NullValueError("Species cannot be null")
        if value not in valid_species:
            raise InvalidCategoryError(f"Species must be one of {valid_species}")
        return True

    @staticmethod
    def validate_culmen_length(value):
        if pd.isna(value):
            raise NullValueError("Culmen length cannot be null")
        try:
            value = float(value)
            if not (30.0 <= value <= 60.0):
                raise ValueRangeError("Culmen length must be between 30.0 and 60.0 mm")
            return True
        except (ValueError, TypeError):
            raise InvalidTypeError("Culmen length must be a number")

    @staticmethod
    def validate_culmen_depth(value):
        if pd.isna(value):
            raise NullValueError("Culmen depth cannot be null")
        try:
            value = float(value)
            if not (13.0 <= value <= 22.0):
                raise ValueRangeError("Culmen depth must be between 13.0 and 22.0 mm")
            return True
        except (ValueError, TypeError):
            raise InvalidTypeError("Culmen depth must be a number")

    @staticmethod
    def validate_flipper_length(value):
        if pd.isna(value):
            raise NullValueError("Flipper length cannot be null")
        try:
            value = float(value)
            if not (170.0 <= value <= 240.0):
                raise ValueRangeError("Flipper length must be between 170.0 and 240.0 mm")
            return True
        except (ValueError, TypeError):
            raise InvalidTypeError("Flipper length must be a number")

    @staticmethod
    def validate_origin_location(value):
        valid_locations = {'Torgersen', 'Biscoe', 'Dream'}
        if pd.isna(value):
            raise NullValueError("Origin location cannot be null")
        if value not in valid_locations:
            raise InvalidCategoryError(f"Origin location must be one of {valid_locations}")
        return True

    @staticmethod
    def validate_body_mass(value):
        if pd.isna(value):
            raise NullValueError("Body mass cannot be null")
        try:
            value = float(value)
            if not (2500.0 <= value <= 6500.0):
                raise ValueRangeError("Body mass must be between 2500.0 and 6500.0 g")
            return True
        except (ValueError, TypeError):
            raise InvalidTypeError("Body mass must be a number")

    @classmethod
    def validate_column(cls, column_name, value):
        validator_map = {
            'Species': cls.validate_species,
            'CulmenLength': cls.validate_culmen_length,
            'CulmenDepth': cls.validate_culmen_depth,
            'FlipperLength': cls.validate_flipper_length,
            'OriginLocation': cls.validate_origin_location,
            'BodyMass': cls.validate_body_mass
        }
        
        validator = validator_map.get(column_name)
        if not validator:
            raise ValueError(f"No validator found for column: {column_name}")
        
        try:
            return validator(value)
        except ValidationError as e:
            raise ValidationError(f"Validation error for {column_name}: {str(e)}")