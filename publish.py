from huggingface_hub import Repository

repo = Repository(local_dir="../", clone_from="Mayura01/T-CLM2")
repo.push_to_hub()
