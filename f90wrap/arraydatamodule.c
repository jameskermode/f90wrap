/*   f90wrap: F90 to Python interface generator with derived type support */

/*   Copyright James Kermode 2011-2018 */

/*   This file is part of f90wrap */
/*   For the latest version see github.com/jameskermode/f90wrap */

/*   f90wrap is free software: you can redistribute it and/or modify */
/*   it under the terms of the GNU Lesser General Public License as published by */
/*   the Free Software Foundation, either version 3 of the License, or */
/*   (at your option) any later version. */

/*   f90wrap is distributed in the hope that it will be useful, */
/*   but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*   GNU Lesser General Public License for more details. */

/*   You should have received a copy of the GNU Lesser General Public License */
/*   along with f90wrap. If not, see <http://www.gnu.org/licenses/>. */
 
/*   If you would like to license the source code under different terms, */
/*   please contact James Kermode, james.kermode@gmail.com */

#include <Python.h>
#include <fortranobject.h>

// http://python3porting.com/cextensions.html
#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif


static PyObject*
get_array(PyObject *self, PyObject *args)
{
  typedef void (*arrayfunc_t)(int*,int*,int*,int*,void*);
  typedef void (*arrayfunc_key_t)(int*,char*,int*,int*,int*,void*,int);

  int nd, i, typenum;
  int dim_temp[10];
  npy_intp *dimensions;
  char *data = NULL;
  PyArrayObject *array = NULL;
  PyArray_Descr *descr = NULL;

  int *this = NULL;
  int sizeof_fortran_t;
  npy_intp this_Dims[1] = {-1};
  const int this_Rank = 1;
  PyArrayObject *capi_this_tmp = NULL;
  int capi_this_intent = 0;
  PyObject *this_capi = NULL;
  PyFortranObject *arrayfunc_capi = NULL;
  char *key = NULL;

  if (!PyArg_ParseTuple(args, "iOO|s", &sizeof_fortran_t,&this_capi,&arrayfunc_capi,&key))
    return NULL;

  /* Processing variable this */
  this_Dims[0]=sizeof_fortran_t;
  capi_this_intent |= F2PY_INTENT_IN;
  capi_this_tmp = array_from_pyobj(PyArray_INT,this_Dims,this_Rank,capi_this_intent,this_capi);
  if (capi_this_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError,"failed in converting 1st argument `this' of get_array to C/Fortran array" );
    goto fail;
  } else {
    this = (int *)(capi_this_tmp->data);
  }

  /* Processing variable arrayfunc */
  if (!PyFortran_Check1(arrayfunc_capi)) {
    PyErr_SetString(PyExc_TypeError, "2nd argument `arrayfunc' is not a fortran object");
    goto fail;
  }
  
  if (arrayfunc_capi->defs[0].rank==-1) {/* is Arrayfunc_Capirtran routine */
    if ((arrayfunc_capi->defs[0].func==NULL)) {
      PyErr_Format(PyExc_RuntimeError, "no function to call");
      goto fail;
    }
    else if (arrayfunc_capi->defs[0].data==NULL) {
      PyErr_Format(PyExc_TypeError, "fortran object is not callable");
      goto fail;
    }
  } else {
    PyErr_Format(PyExc_TypeError, "fortran object is not callable");
    goto fail;
  }

  /* Call arrayfunc_capi routine */
  if (key == NULL) 
    ((arrayfunc_t)(arrayfunc_capi->defs[0].data))(this, &nd, &typenum, dim_temp, &data);
  else
    ((arrayfunc_key_t)(arrayfunc_capi->defs[0].data))(this, key, &nd, &typenum, dim_temp, &data, strlen(key));

  if (data == NULL) {
    PyErr_SetString(PyExc_ValueError, "array is NULL");
    goto fail;
  }

  dimensions = (npy_intp*)malloc(nd*sizeof(npy_intp));
  for (i=0; i<nd; i++) {
    dimensions[i] = (npy_intp)(dim_temp[i]);
  }

  /* Construct array */
  descr = PyArray_DescrNewFromType(typenum);
  array = (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, descr, nd, dimensions, NULL, 
                                                data, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED, NULL);
  free(dimensions);
  if((PyObject *)capi_this_tmp!=this_capi) {
    Py_XDECREF(capi_this_tmp);
  }
  return (PyObject *)array;

 fail:
  Py_XDECREF(descr);
  if(capi_this_tmp != NULL && ((PyObject *)capi_this_tmp!=this_capi)) {
    Py_XDECREF(capi_this_tmp);
  }
  return NULL;
}


static PyMethodDef arraydata_methods[] = {
  {"get_array", get_array, METH_VARARGS, 
   "Make an array from integer(sizeof_fortran_t) array containing reference to derived type object,\n and fortran array function.\n\get_array(sizeof_fortran_t, fpointer,array_fobj[,key]) -> array"},
  {NULL, NULL}
};

static char arraydata_doc[] = 
  "Extension module to create numpy arrays which access existing data at a given memory location";


// =====================================================
#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int arraydataTraverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int arraydataClear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static PyModuleDef arraydataModuleDef = {
    PyModuleDef_HEAD_INIT,
    "arraydata",
    arraydata_doc,
    sizeof(struct module_state),
    arraydata_methods,
    NULL,
    arraydataTraverse,
    arraydataClear,
    NULL
};


PyMODINIT_FUNC
PyInit_arraydata(void)
{
  PyObject *mod = PyModule_Create(&arraydataModuleDef);
#if PY_MINOR_VERSION >= 11 // https://peps.python.org/pep-0674/#port-c-extensions-to-python-3-11
  Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
#else
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
#endif

  import_array();
  return mod;
}



// =====================================================
#else

PyMODINIT_FUNC
initarraydata(void)
{
  Py_InitModule3("arraydata", arraydata_methods, arraydata_doc);
  PyFortran_Type.ob_type = &PyType_Type;
  import_array();
}

#endif

// =====================================================

