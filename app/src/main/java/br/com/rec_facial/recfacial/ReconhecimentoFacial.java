package br.com.rec_facial.recfacial;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.ExifInterface;
import android.media.Image;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.Pair;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import br.com.rec_facial.recfacial.SimilarityClassifier;

public class ReconhecimentoFacial  extends AppCompatActivity {

    String modelFile = "mobile_face_net.tflite";

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    CameraSelector cameraSelector;
    boolean start = true, flipX = false;
    int cam_face = CameraSelector.LENS_FACING_FRONT;

    int[] intValues;
    int inputSize = 112;
    boolean isModelQuantized = false;
    float[][] embeedings;
    float IMAGE_MEAN = 128.0f;
    float IMAGE_STD = 128.0f;
    int OUTPUT_SIZE = 192;
    private static int SELECT_PICTURE = 1;
    ProcessCameraProvider cameraProvider;
    private static final int MY_CAMERA_REQUEST_CODE = 100;

    //adiciona uma face para detecção
    public void addFace(Context context, float[][] embeedings, HashMap<String, SimilarityClassifier.Recognition> registered ) {

    }


    public MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Boolean recognizeImage(final Bitmap bitmap,  HashMap<String, SimilarityClassifier.Recognition> registered, TextView reco_name, Interpreter tfLite ) {

        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4);

        imgData.order(ByteOrder.nativeOrder());

        intValues = new int[inputSize * inputSize];

        //get pixel values from Bitmap to normalize
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

                }
            }
        }
        //imgData is input to our model
        Object[] inputArray = {imgData};

        Map<Integer, Object> outputMap = new HashMap<>();


        embeedings = new float[1][OUTPUT_SIZE]; //output of model will be stored in this variable

        outputMap.put(0, embeedings);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap); //Run model


        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

        //Compare new face with saved Faces.
        if (registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(embeedings[0], registered);
            if (nearest != null) {
                final String name = nearest.first;
                label = name;
                distance = nearest.second;
                if (distance < 1.000f){
                    if(reco_name != null){
                        reco_name.setText(name);
                        return true;
                    }
                }else{
                    if(reco_name != null) {
                        reco_name.setText("Não Reconhecida");
                        return false;
                    }
                }
                System.out.println("nearest: " + name + " - distance: " + distance);
            }
        }

        return false;
    }


    //Compare Faces by distance between face embeddings
    private Pair<String, Float> findNearest(float[] emb, HashMap<String, SimilarityClassifier.Recognition> registered ) {

        Pair<String, Float> ret = null;
        if(registered.size() > 0 && registered.get(0) != null){
            for (Map.Entry<String, SimilarityClassifier.Recognition> entry : registered.entrySet()) {

                final String name = entry.getKey();
                final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

                float distance = 0;
                for (int i = 0; i < emb.length; i++) {
                    float diff = emb[i] - knownEmb[i];
                    distance += diff * diff;
                }
                distance = (float) Math.sqrt(distance);
                if (ret == null || distance < ret.second) {
                    ret = new Pair<>(name, distance);
                }
            }
        }
        return ret;
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    public static Bitmap getCropBitmapByCPU(Bitmap source, RectF cropRectF) {
        Bitmap resultBitmap = Bitmap.createBitmap((int) cropRectF.width(),
                (int) cropRectF.height(), Bitmap.Config.ARGB_8888);
        Canvas cavas = new Canvas(resultBitmap);

        // draw background
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);
        cavas.drawRect(//from  w w  w. ja v  a  2s. c  om
                new RectF(0, 0, cropRectF.width(), cropRectF.height()),
                paint);

        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left, -cropRectF.top);

        cavas.drawBitmap(source, matrix, paint);

        if (source != null && !source.isRecycled()) {
            source.recycle();
        }

        return resultBitmap;
    }

    public static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX, boolean flipY) {
        Matrix matrix = new Matrix();

        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);

        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, flipY ? -1.0f : 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }


    public static byte[] YUV_420_888toNV21(Image image) {

        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        assert (image.getPlanes()[0].getPixelStride() == 1);

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        } else {
            long yBufferPos = -rowStride; // not an actual position
            for (; pos < ySize; pos += width) {
                yBufferPos += rowStride;
                yBuffer.position((int) yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        assert (rowStride == image.getPlanes()[1].getRowStride());
        assert (pixelStride == image.getPlanes()[1].getPixelStride());

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            byte savePixel = vBuffer.get(1);
            try {
                vBuffer.put(1, (byte) ~savePixel);
                if (uBuffer.get(0) == (byte) ~savePixel) {
                    vBuffer.put(1, savePixel);
                    vBuffer.position(0);
                    uBuffer.position(0);
                    vBuffer.get(nv21, ySize, 1);
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining());

                    return nv21; // shortcut
                }
            } catch (ReadOnlyBufferException ex) {
                // unfortunately, we cannot check if vBuffer and uBuffer overlap
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel);
        }

        // other optimizations could check if (pixelStride == 1) or (pixelStride == 2),
        // but performance gain would be less significant

        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int vuPos = col * pixelStride + row * rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        return nv21;
    }

    public Bitmap toBitmap(Image image) {

        byte[] nv21 = YUV_420_888toNV21(image);


        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    //Save Faces to Shared Preferences.Conversion of Recognition objects to json string
    public void insertToSP(SharedPreferences sharedPrefe, Context context, HashMap<String, SimilarityClassifier.Recognition> jsonMap, boolean clear) {
        if (clear)
            jsonMap.clear();
        else
            jsonMap.putAll(readFromSP(sharedPrefe));
        String jsonString = new Gson().toJson(jsonMap);
        SharedPreferences sharedPreferences = getSharedPreferences("HashMap", MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("map", jsonString);
        editor.apply();
        Toast.makeText(context, "Recognitions Saved", Toast.LENGTH_SHORT).show();
    }

    //Carregar Faces de String compartilhada Preferences.Json para objeto de reconhecimento
    public HashMap<String, SimilarityClassifier.Recognition> readFromSP(SharedPreferences sharedPreferences) {
        String defValue = new Gson().toJson(new HashMap<String, SimilarityClassifier.Recognition>());
        String json = sharedPreferences.getString("map", defValue);
        TypeToken<HashMap<String, SimilarityClassifier.Recognition>> token = new TypeToken<HashMap<String, SimilarityClassifier.Recognition>>() {
        };
        HashMap<String, SimilarityClassifier.Recognition> retrievedMap = new Gson().fromJson(json, token.getType());
        for (Map.Entry<String, SimilarityClassifier.Recognition> entry : retrievedMap.entrySet()) {
            float[][] output = new float[1][OUTPUT_SIZE];
            ArrayList arrayList = (ArrayList) entry.getValue().getExtra();
            arrayList = (ArrayList) arrayList.get(0);
            for (int counter = 0; counter < arrayList.size(); counter++) {
                output[0][counter] = ((Double) arrayList.get(counter)).floatValue();
            }
            entry.getValue().setExtra(output);
        }
        return retrievedMap;
    }

    //Carregar foto do armazenamento do telefone
    private void loadphoto() {
        start = false;
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Selecione uma Foto"), SELECT_PICTURE);
    }

    public  Bitmap getBitmapFromUri(Uri uri) throws IOException {
        ParcelFileDescriptor parcelFileDescriptor = getContentResolver().openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();
        return image;
    }

    public void capturarFotoDefault(Context context, String localFoto, FaceDetector detector,
                                    float[][] embeedings, HashMap<String, SimilarityClassifier.Recognition> registered,  Interpreter tfLite ) {
        try {
            ajustarResolucaoImagem(localFoto);
            Bitmap bitmap = rotateImage(BitmapFactory.decodeFile(localFoto), localFoto);
            InputImage impphoto = InputImage.fromBitmap(bitmap, 0);
            detector.process(impphoto).addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                @Override
                public void onSuccess(List<Face> faces) {

                    if (faces.size() != 0) {
                        Face face = faces.get(0);
                        System.out.println(face);

                        Bitmap frame_bmp = null;
                        frame_bmp = BitmapFactory.decodeFile(localFoto);

                        Bitmap frame_bmp1 = bitmap;//rotateBitmap(frame_bmp, 0, flipX, false);
                        RectF boundingBox = new RectF(face.getBoundingBox());
                        Bitmap cropped_face = getCropBitmapByCPU(frame_bmp1, boundingBox);
                        Bitmap scaled = getResizedBitmap(cropped_face, 112, 112);

                        recognizeImage(scaled, registered, null, tfLite);
                        adicionarFaceDefault(embeedings, registered);
                        System.out.println(boundingBox);
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    Toast.makeText(context, "Failed to add", Toast.LENGTH_SHORT).show();
                }
            });
            //  face_preview.setImageBitmap(bitmap);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    //adiciona uma face default
    public void adicionarFaceDefault(float[][] embeedings, HashMap<String, SimilarityClassifier.Recognition> registered) {
        {
            SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                    "0", "", -1f);
            result.setExtra(embeedings);
            registered.put("evalton", result);
        }
    }

    public Bitmap rotateImage(Bitmap bitmap, String path) {
        ExifInterface exifInterface = null;
        try {
            exifInterface = new ExifInterface(path);
        } catch (IOException e) {
            Log.e("IOException", e.getMessage(), e);
        }

        int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case 8:
                matrix.setRotate(270);
                break;
            case 0:
                matrix.setRotate(90);
                break;
            default:
        }
        Bitmap rotateBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return rotateBitmap;
    }

    public void ajustarResolucaoImagem(String imagePath) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 4;

            Bitmap bitmap = BitmapFactory.decodeStream(new FileInputStream(imagePath), null, null);
            if (verificaNecessidadeRedimencionar(bitmap)) {
                bitmap = BitmapFactory.decodeStream(new FileInputStream(imagePath), null, options);
            }
            String[] nome = imagePath.split("/");
            String filename = nome[3];
            String path = nome[0] + "/" + nome[1] + "/" + nome[2] + "/" + nome[3] + "/" + nome[4] + "/" + nome[5] + "/" + nome[6] + "/" + nome[7] + "/" + nome[8] + "/";

            File file = null;
            file = new File(path, filename);
            if (file != null) {
                file.delete();
                file = new File(path, filename);
            }
            FileOutputStream fos;
            try {
                fos = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 50, fos);
                fos.flush();
                fos.close();
            } catch (FileNotFoundException e) {
                Log.e("Filenotfound", e.getMessage(), e);
            } catch (IOException e) {
                Log.e("IOException", e.getMessage(), e);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    private Boolean verificaNecessidadeRedimencionar(Bitmap bitmap) {
        Integer tamanho = bitmap.getHeight() * bitmap.getWidth();
        return tamanho > 300000;
    }

    /*
    *
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                try {
                    InputImage impphoto = InputImage.fromBitmap(reconhecimentoFacial.getBitmapFromUri(selectedImageUri), 0);
                    detector.process(impphoto).addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                        @Override
                        public void onSuccess(List<Face> faces) {

                            if (faces.size() != 0) {
                              //  iniciar.setText("Reconhecimento");
                              //  iniciar.setVisibility(View.VISIBLE);
                                face_preview.setVisibility(View.VISIBLE);
                                Face face = faces.get(0);
                                System.out.println(face);

                                Bitmap frame_bmp = null;
                                try {
                                    frame_bmp = reconhecimentoFacial.getBitmapFromUri(selectedImageUri);
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                                Bitmap frame_bmp1 = reconhecimentoFacial.rotateBitmap(frame_bmp, 0, flipX, false);
                                RectF boundingBox = new RectF(face.getBoundingBox());

                                Bitmap cropped_face = reconhecimentoFacial.getCropBitmapByCPU(frame_bmp1, boundingBox);

                                Bitmap scaled = reconhecimentoFacial.getResizedBitmap(cropped_face, 112, 112);

                                recognizeImage(scaled);
                                reconhecimentoFacial.addFace(context,embeedings, registered);
                                System.out.println(boundingBox);
                                try {
                                    Thread.sleep(100);
                                } catch (InterruptedException e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                    }).addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            start = true;
                            Toast.makeText(context, "Failed to add", Toast.LENGTH_SHORT).show();
                        }
                    });
                    face_preview.setImageBitmap(reconhecimentoFacial.getBitmapFromUri(selectedImageUri));
                } catch (IOException e) {
                    e.printStackTrace();
                }


            }
        }
    }

    * */
}
