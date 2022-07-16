SPECIAL_ENTITY_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

def get_relation_labels(num_labels):
    return [f"[LABEL{i}]" for i in range(1, num_labels + 1)]