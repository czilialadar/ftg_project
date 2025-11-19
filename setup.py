from setuptools import find_packages, setup

package_name = 'ftg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ftg_labirinth.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/ftg_box.launch.xml']),
        ('share/' + package_name + '/rviz', ['rviz/ftg.rviz'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eddy',
    maintainer_email='eddy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [ 
            'ftg = ftg.ftg:main'
        ],
    },
)
