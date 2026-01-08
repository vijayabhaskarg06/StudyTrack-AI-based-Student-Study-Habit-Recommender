from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Get an item from a dictionary using a key.
    Usage: {{ mydict|get_item:"keyname" }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''

@register.filter
def get_attr(obj, attr_name):
    """Get attribute from object or dictionary."""
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    elif isinstance(obj, dict) and attr_name in obj:
        return obj[attr_name]
    return ''