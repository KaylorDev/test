def test_split():
    from src.file_parser import split_text_into_chunks
    
    text = (
        "— Где это я? — пробормотал я, оглядываясь. Вместо привычного потолка "
        "я увидел деревянные балки. Запах был странный. "
        "Казалось, что я попал в другой мир. "
        "Ну и дела! " * 20 # Make it long
    )
    
    print(f"Original text length: {len(text)}")
    chunks = split_text_into_chunks(text, max_chars=400)
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} length: {len(c)}")
        print(f"Chunk {i} start: {c[:20]}...")

if __name__ == "__main__":
    test_split()
