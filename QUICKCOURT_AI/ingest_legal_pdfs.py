import os
import re
import json
import hashlib

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
TRACK_FILE = "processed_files.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

BATCH_SIZE = 100


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# LOAD TRACKING FILE
# ============================================================

if os.path.exists(TRACK_FILE):

    with open(
        TRACK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        processed_files = json.load(f)

else:

    processed_files = {}


# ============================================================
# SAVE TRACKING FILE
# ============================================================

def save_tracking():

    with open(
        TRACK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            processed_files,
            f,
            indent=4
        )


# ============================================================
# SECTION DETECTION
# ============================================================

SECTION_REGEX = re.compile(
    r"(?m)^\s*(\d+[A-Z]?)\.\s+(.+)$"
)


# ============================================================
# CLEAN SECTION TITLE
# ============================================================

def clean_section_title(title):

    title = title.strip()

    # Remove text after em dash
    if "—" in title:

        title = title.split(
            "—",
            1
        )[0]

    # Remove text after double hyphen
    if "--" in title:

        title = title.split(
            "--",
            1
        )[0]

    # Remove trailing punctuation
    title = title.strip(
        " .:-"
    )

    return title.strip()


# ============================================================
# POSSIBLE TABLE OF CONTENTS DETECTION
# ============================================================

def is_toc_line(line):

    line = line.strip()

    if not line:
        return False

    # Example:
    #
    # 103. Punishment for murder ........ 47

    pattern = re.compile(
        r"^\d+[A-Z]?\.\s+.+\.{2,}\s*\d+\s*$"
    )

    return bool(
        pattern.match(line)
    )


# ============================================================
# CREATE STABLE UNIQUE CHUNK ID
# ============================================================

def create_chunk_id(
    document_id,
    page_number,
    section,
    chunk_number,
    text
):

    raw = (
        f"{document_id}|"
        f"{page_number}|"
        f"{section}|"
        f"{chunk_number}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# FIND ALL PDF FILES
# ============================================================

pdf_files = sorted(

    [
        file
        for file in os.listdir(
            DATA_FOLDER
        )

        if file.lower().endswith(".pdf")
    ]

)


if not pdf_files:

    raise FileNotFoundError(
        "No PDF files found inside data folder."
    )


print("\n========================================")
print("QUICKCOURT AI — LEGAL PDF INGESTION")
print("========================================")

print(
    f"PDF files found: {len(pdf_files)}"
)

for pdf in pdf_files:

    print(
        " •",
        pdf
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)


embeddings = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"

)


print(
    "Embedding model loaded."
)


# ============================================================
# CONNECT TO CHROMA
# ============================================================

vector_db = Chroma(

    persist_directory=DB_FOLDER,

    embedding_function=embeddings

)


# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP

)


# ============================================================
# PROCESS EACH PDF
# ============================================================

for pdf in pdf_files:

    pdf_path = os.path.join(
        DATA_FOLDER,
        pdf
    )


    print(
        "\n----------------------------------------"
    )

    print(
        f"Checking: {pdf}"
    )


    # ========================================================
    # CURRENT PDF HASH
    # ========================================================

    current_hash = get_file_hash(
        pdf_path
    )


    # ========================================================
    # DOCUMENT ID
    # ========================================================

    # Based on filename.
    #
    # Example:
    # BNS.pdf -> BNS
    # IPC.pdf -> IPC

    document_id = os.path.splitext(
        pdf
    )[0].lower()


    # ========================================================
    # CHECK PREVIOUS VERSION
    # ========================================================

    previous_record = (
        processed_files.get(
            pdf
        )
    )


    # ========================================================
    # PDF UNCHANGED
    # ========================================================

    if (

        previous_record

        and

        previous_record.get(
            "pdf_hash"
        )
        == current_hash

    ):

        print(
            f"⏭ Skipped: {pdf}"
        )

        print(
            "   Reason: PDF unchanged."
        )

        continue


    # ========================================================
    # PDF IS NEW OR MODIFIED
    # ========================================================

    if previous_record:

        print(
            "🔄 PDF changed."
        )

        print(
            "   Old vectors will be removed."
        )

    else:

        print(
            "🆕 New PDF detected."
        )


    # ========================================================
    # DELETE OLD VECTORS
    # ========================================================

    if previous_record:

        try:

            vector_db._collection.delete(

                where={
                    "document_id":
                        document_id
                }

            )

            print(
                "   ✅ Old vectors deleted."
            )

        except Exception as e:

            print(
                "   ⚠ Could not delete old vectors:"
            )

            print(
                "   ",
                e
            )

            raise


    # ========================================================
    # LOAD PDF
    # ========================================================

    print(
        "   Loading PDF..."
    )


    loader = PyPDFLoader(
        pdf_path
    )

    pages = loader.load()


    print(
        f"   Pages: {len(pages)}"
    )


    # ========================================================
    # SECTION STATE
    # ========================================================

    current_section = "Unknown"

    current_title = "Unknown"


    pdf_chunks = []

    pdf_ids = []


    # ========================================================
    # PROCESS PAGES
    # ========================================================

    for page_index, page in enumerate(
        pages
    ):

        text = page.page_content.strip()


        if not text:

            continue


        # ----------------------------------------------------
        # Page information
        # ----------------------------------------------------

        page_number = page.metadata.get(
            "page",
            page_index
        )


        page_label = page.metadata.get(
            "page_label",
            str(page_number)
        )


        # ----------------------------------------------------
        # Find section headings
        # ----------------------------------------------------

        matches = list(
            SECTION_REGEX.finditer(
                text
            )
        )


        # ====================================================
        # NO SECTION HEADING
        # → continuation page
        # ====================================================

        if not matches:

            if current_section == "Unknown":

                # Ignore front matter
                continue


            doc = Document(

                page_content=text,

                metadata={

                    "document_id":
                        document_id,

                    "pdf_hash":
                        current_hash,

                    "law":
                        document_id.upper(),

                    "section":
                        current_section,

                    "section_title":
                        current_title,

                    "page":
                        str(page_label),

                    "source":
                        pdf,

                    "content_type":
                        "section",

                    "is_continuation":
                        True

                }

            )


            docs = splitter.split_documents(
                [doc]
            )


            for chunk_index, chunk in enumerate(
                docs
            ):

                chunk_id = create_chunk_id(

                    document_id,

                    page_number,

                    current_section,

                    f"continuation_{chunk_index}",

                    chunk.page_content

                )


                pdf_chunks.append(
                    chunk
                )

                pdf_ids.append(
                    chunk_id
                )


            continue


        # ====================================================
        # PAGE HAS SECTION(S)
        # ====================================================

        for match_index, match in enumerate(
            matches
        ):


            # ------------------------------------------------
            # Section number
            # ------------------------------------------------

            section_number = match.group(
                1
            )


            # ------------------------------------------------
            # Raw title
            # ------------------------------------------------

            raw_title = match.group(
                2
            ).strip()


            # ------------------------------------------------
            # Skip obvious TOC lines
            # ------------------------------------------------

            if is_toc_line(
                match.group(0)
            ):

                continue


            section = (
                f"Section {section_number}"
            )


            title = clean_section_title(
                raw_title
            )


            # ------------------------------------------------
            # Text BEFORE first section
            #
            # This can be continuation text.
            # ------------------------------------------------

            if match_index == 0:

                before_text = (
                    text[:match.start()]
                    .strip()
                )


                if (

                    before_text

                    and

                    current_section
                    != "Unknown"

                ):

                    continuation_doc = Document(

                        page_content=
                            before_text,

                        metadata={

                            "document_id":
                                document_id,

                            "pdf_hash":
                                current_hash,

                            "law":
                                document_id.upper(),

                            "section":
                                current_section,

                            "section_title":
                                current_title,

                            "page":
                                str(page_label),

                            "source":
                                pdf,

                            "content_type":
                                "section",

                            "is_continuation":
                                True

                        }

                    )


                    continuation_chunks = (
                        splitter.split_documents(
                            [continuation_doc]
                        )
                    )


                    for chunk_index, chunk in enumerate(
                        continuation_chunks
                    ):


                        chunk_id = (
                            create_chunk_id(

                                document_id,

                                page_number,

                                current_section,

                                f"before_{chunk_index}",

                                chunk.page_content

                            )
                        )


                        pdf_chunks.append(
                            chunk
                        )

                        pdf_ids.append(
                            chunk_id
                        )


            # ------------------------------------------------
            # End of current section
            # ------------------------------------------------

            if match_index + 1 < len(matches):

                end = matches[
                    match_index + 1
                ].start()

            else:

                end = len(text)


            section_text = (
                text[
                    match.start():end
                ].strip()
            )


            if not section_text:

                continue


            # ------------------------------------------------
            # Update current section
            # ------------------------------------------------

            current_section = section

            current_title = title


            # ------------------------------------------------
            # Create Document
            # ------------------------------------------------

            doc = Document(

                page_content=
                    section_text,

                metadata={

                    "document_id":
                        document_id,

                    "pdf_hash":
                        current_hash,

                    "law":
                        document_id.upper(),

                    "section":
                        section,

                    "section_title":
                        title,

                    "page":
                        str(page_label),

                    "source":
                        pdf,

                    "content_type":
                        "section",

                    "is_continuation":
                        False

                }

            )


            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            docs = splitter.split_documents(
                [doc]
            )


            # ------------------------------------------------
            # Create IDs
            # ------------------------------------------------

            for chunk_index, chunk in enumerate(
                docs
            ):


                chunk_id = (
                    create_chunk_id(

                        document_id,

                        page_number,

                        section,

                        f"{match_index}_{chunk_index}",

                        chunk.page_content

                    )
                )


                pdf_chunks.append(
                    chunk
                )

                pdf_ids.append(
                    chunk_id
                )


    # ========================================================
    # VALIDATION
    # ========================================================

    if len(pdf_chunks) != len(pdf_ids):

        raise RuntimeError(
            "Document and ID count mismatch."
        )


    if len(pdf_ids) != len(
        set(pdf_ids)
    ):

        raise RuntimeError(
            "Duplicate IDs generated."
        )


    print(
        f"   Chunks generated: "
        f"{len(pdf_chunks)}"
    )


    # ========================================================
    # INSERT INTO CHROMA
    # ========================================================

    if pdf_chunks:

        print(
            "   Adding vectors..."
        )


        for start in range(
            0,
            len(pdf_chunks),
            BATCH_SIZE
        ):

            end = min(

                start + BATCH_SIZE,

                len(pdf_chunks)

            )


            vector_db.add_documents(

                documents=
                    pdf_chunks[
                        start:end
                    ],

                ids=
                    pdf_ids[
                        start:end
                    ]

            )


            print(

                f"   Added "
                f"{end}/"
                f"{len(pdf_chunks)}"

            )


        # ====================================================
        # ONLY AFTER SUCCESS
        # UPDATE TRACKING
        # ====================================================

        processed_files[pdf] = {

            "document_id":
                document_id,

            "pdf_hash":
                current_hash,

            "chunks":
                len(pdf_chunks)

        }


        save_tracking()


        print(
            f"   ✅ Successfully indexed: {pdf}"
        )


    else:

        print(
            f"   ⚠ No chunks generated: {pdf}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "QUICKCOURT AI KNOWLEDGE BASE READY"
)

print(
    "========================================"
)

print(
    "Tracked PDFs:",
    len(processed_files)
)

print(
    "Database:",
    DB_FOLDER
)

print(
    "Tracking file:",
    TRACK_FILE
)

print(
    "========================================"
)

# NEXT STEP:
# Build a production-quality test_retrieval.py / Legal Retrieval Engine that:
# - searches ChromaDB,
# - groups/handles section continuations,
# - removes duplicate results,
# - returns structured legal sources,
# - and prepares grounded context for the LLM.