import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final ImagePicker _picker = ImagePicker();
  File? _image;
  late Interpreter _interpreter;
  final List<String> _classNames = [
    "Premature (0-2)",
    "Premature (3-4)",
    "Premature (5-7)",
    "Mature (8)",
    "Mature (9)",
    "Mature (10)",
    "Over-Mature (11)",
    "Over-Mature (12)",
    "Over-Mature (13 or above)"
  ];

  String? _predictedLabel;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model_unquant.tflite');
      if (_interpreter != null) {
        _interpreter!.resizeInputTensor(0, [1, 256, 256, 3]);
        _interpreter!.allocateTensors();
      }
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  Future<void> _getImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
        classifyImage(_image!);
      } else {
        print('No image selected.');
      }
    });
  }

  Future<void> classifyImage(File imageFile) async {
    try {
      Uint8List imageBytes = await imageFile.readAsBytes();
      img_lib.Image? image = img_lib.decodeImage(imageBytes);
      if (image == null) {
        print('Error decoding image.');
        return;
      }

      img_lib.Image resizedImage = img_lib.copyResize(image, width: 256, height: 256);

      final inputBuffer = Float32List(256 * 256 * 3);

      for (var i = 0; i < 256; i++) {
        for (var j = 0; j < 256; j++) {
          var pixel = resizedImage.getPixel(j, i);
          inputBuffer[i * 256 * 3 + j * 3 + 0] = ((pixel.r.toDouble() - 127.5) / 127.5).toDouble();
          inputBuffer[i * 256 * 3 + j * 3 + 1] = ((pixel.g.toDouble() - 127.5) / 127.5).toDouble();
          inputBuffer[i * 256 * 3 + j * 3 + 2] = ((pixel.b.toDouble() - 127.5) / 127.5).toDouble();
        }
      }

      final outputBuffer = Float32List(1 * 9);
      _interpreter.run(inputBuffer.buffer.asUint8List(), outputBuffer.buffer.asUint8List());

      final double maxConfidence = outputBuffer.reduce((a, b) => a > b ? a : b);
      final int index = outputBuffer.indexOf(maxConfidence);
      final String label = _classNames[index];

      setState(() {
        _predictedLabel = label;
      });

      print('Label: $label');
    } catch (error) {
      print('Error resizing and preprocessing image: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? const Text('No image selected.')
                : Column(
              children: [
                Image.file(_image!),
                const SizedBox(height: 16),
                Text('Predicted Label: $_predictedLabel'),
              ],
            ),
          ],
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: () => _getImage(ImageSource.gallery),
            tooltip: 'Pick Image from Gallery',
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(width: 16),
          FloatingActionButton(
            onPressed: () => _getImage(ImageSource.camera),
            tooltip: 'Take a Photo',
            child: const Icon(Icons.camera_alt),
          ),
        ],
      ),
    );
  }
}

