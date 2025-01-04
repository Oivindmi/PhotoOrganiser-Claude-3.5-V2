from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import csv
import json
import logging

Base = declarative_base()

class FileMetadata(Base):
    __tablename__ = 'file_metadata'

    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    file_info = Column(String, unique=True)
    camera_model = Column(String)
    file_size = Column(Float)
    extra_metadata = Column(JSON)


class DatabaseManager:
    def __init__(self, db_path='photo_organizer.db'):
        self.logger = logging.getLogger(__name__)
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.Session = sessionmaker(bind=self.engine)
        self.recreate_tables()

    def recreate_tables(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.logger.info("Database tables recreated successfully.")

    def clear_database(self):
        self.recreate_tables()

    def add_file_metadata(self, metadata):
        session = self.Session()
        try:
            file_info = metadata.get('file_info')
            file_path = metadata.get('file_path')
            camera_model = metadata.get('camera_model')
            file_size = metadata.get('file_size')

            extra_metadata = metadata.get('extra_metadata', {})

            existing_record = session.query(FileMetadata).filter_by(file_info=file_info).first()
            if existing_record:
                existing_record.file_path = file_path
                existing_record.camera_model = camera_model
                existing_record.file_size = file_size
                existing_record.extra_metadata = extra_metadata
            else:
                file_metadata = FileMetadata(
                    file_path=file_path,
                    file_info=file_info,
                    camera_model=camera_model,
                    file_size=file_size,
                    extra_metadata=extra_metadata
                )
                session.add(file_metadata)
            session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error adding file metadata: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    def export_to_csv(self, file_path):
        session = self.Session()
        try:
            data = session.query(FileMetadata).all()

            if not data:
                self.logger.warning("No data to export.")
                return False

            fieldnames = ['id', 'file_path', 'file_info', 'camera_model', 'file_size']
            all_metadata_keys = set()
            for item in data:
                all_metadata_keys.update(item.extra_metadata.keys())
            fieldnames.extend(sorted(all_metadata_keys))

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in data:
                    row = {
                        'id': item.id,
                        'file_path': item.file_path,
                        'file_info': item.file_info,
                        'camera_model': item.camera_model,
                        'file_size': item.file_size,
                    }
                    row.update(item.extra_metadata)
                    writer.writerow(row)

            self.logger.info(f"Database exported to CSV: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting database to CSV: {str(e)}")
            return False
        finally:
            session.close()

    def view_database_contents(self):
        session = self.Session()
        try:
            total_entries = session.query(FileMetadata).count()
            sample_entries = session.query(FileMetadata).limit(5).all()

            print(f"\nTotal entries in database: {total_entries}")
            print("\nSample entries:")
            for entry in sample_entries:
                print(
                    f"  - {entry.file_path} (Camera: {entry.camera_model or 'Unknown'}, Size: {entry.file_size or 'Unknown'} MB)")
                print(f"    Extra Metadata: {json.dumps(entry.extra_metadata, indent=2)}")

        except Exception as e:
            self.logger.error(f"Error viewing database contents: {str(e)}")
        finally:
            session.close()

    def get_all_file_metadata(self):
        session = self.Session()
        try:
            return session.query(FileMetadata).all()
        except Exception as e:
            self.logger.error(f"Error retrieving all file metadata: {str(e)}")
            return []
        finally:
            session.close()