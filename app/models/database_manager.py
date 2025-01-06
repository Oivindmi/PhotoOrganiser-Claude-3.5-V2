from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Table, MetaData, text, inspect, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import json

Base = declarative_base()

class FileMetadata(Base):
    __tablename__ = 'file_metadata'

    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    file_info = Column(String, unique=True)
    camera_model = Column(String)
    file_size = Column(Float)
    extra_metadata = Column(JSON)
    correct_time = Column(DateTime)
    original_time_field = Column(String)
    group_id = Column(Integer)
    group_key = Column(String)
    video_frames = Column(JSON)
    is_video = Column(Boolean, default=False)


class DatabaseManager:
    def __init__(self, db_path='photo_organizer.db'):
        self.logger = logging.getLogger(__name__)
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.Session = sessionmaker(bind=self.engine)
        self.FileMetadata = FileMetadata
        self.create_tables()

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        self.logger.info("Database tables created.")

        inspector = inspect(self.engine)
        for table_name in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            self.logger.info(f"Table {table_name} columns: {', '.join(columns)}")

    def add_file_metadata_bulk(self, metadata_list):
        session = self.Session()
        try:
            for item in metadata_list:
                extra_metadata = item.get('extra_metadata', {})
                if isinstance(extra_metadata, str):
                    try:
                        extra_metadata = json.loads(extra_metadata)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse extra_metadata JSON for file: {item.get('file_path')}")
                        extra_metadata = {}

                file_metadata = FileMetadata(
                    file_path=item['file_path'],
                    file_info=item['file_info'],
                    camera_model=item['camera_model'],
                    file_size=item['file_size'],
                    extra_metadata=extra_metadata,
                    correct_time=item.get('correct_time'),
                    original_time_field=item.get('original_time_field', ''),
                    group_id=item.get('group_id'),
                    group_key=item.get('group_key'),
                    is_video=item.get('is_video', False),
                    video_frames=item.get('video_frames')
                )
                session.merge(file_metadata)
            session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error adding file metadata in bulk: {str(e)}")
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

            fieldnames = ['id', 'file_path', 'file_info', 'camera_model', 'file_size', 'extra_metadata']

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
                        'extra_metadata': item.extra_metadata
                    }
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
                print(f"  - {entry.file_path} (Camera: {entry.camera_model or 'Unknown'}, Size: {entry.file_size or 'Unknown'} MB)")
                print(f"    Extra Metadata: {entry.extra_metadata}")

        except Exception as e:
            self.logger.error(f"Error viewing database contents: {str(e)}")
        finally:
            session.close()

    def clear_cache(self):
        self.Session.close_all()
        self.engine.dispose()

    def update_group_time(self, session, group_id, time_delta):
        try:
            session.query(self.FileMetadata).filter_by(group_id=group_id).update(
                {self.FileMetadata.correct_time: self.FileMetadata.correct_time + time_delta}
            )
            session.commit()
        except Exception as e:
            self.logger.error(f"Error updating group time: {str(e)}")
            session.rollback()