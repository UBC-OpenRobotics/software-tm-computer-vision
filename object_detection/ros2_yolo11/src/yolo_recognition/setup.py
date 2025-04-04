from setuptools import find_packages, setup

package_name = 'yolo_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kevin',
    maintainer_email='kevinleimc@gmail.com',
    description='Open Robotics Yolo Detection Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = yolo_recognition.webcam_publisher:main',
            'listener = yolo_recognition.yolo_subscriber:main',
        ],
    },
)
