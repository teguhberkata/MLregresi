package com.example.mlregresi

import android.content.res.AssetManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import org.w3c.dom.Text
import java.io.FileInputStream
import java.io.InputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.channels.FileChannel.MapMode
import java.nio.file.Path

class MainActivity : AppCompatActivity(){

    private lateinit var interpreter: Interpreter
    private val mModelPath = "linear.tflite"

    private lateinit var resultText: TextView
    private lateinit var editText: EditText
    private lateinit var checkButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.txtResult)
        editText = findViewById(R.id.numberField)
        checkButton = findViewById(R.id.btnPredict)

        checkButton.setOnClickListener{
            var result = doInference(editText.text.toString())
            runOnUiThread{
                resultText.text = result.toString()
            }
        }

        initInterpreter()
    }

    private fun initInterpreter(){
        val options = Interpreter.Options()
        options.setNumThreads(4)
        options.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assets, mModelPath), options)
    }

    private fun doInference(inputString: String): Float{
        val inputVal = FloatArray(1)
        inputVal[0] = inputString.toFloat()
        val ouput = Array(1) {FloatArray(1)}
        interpreter.run(inputVal, ouput)
        return ouput[0][0]
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer{
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

}