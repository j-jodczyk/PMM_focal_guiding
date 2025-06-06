
"""SCons.Tool.qt

Tool-specific initialization for Qt.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__revision__ = "/home/scons/scons/branch.0/branch.96/baseline/src/engine/SCons/Tool/qt.py 0.96.92.D001 2006/04/10 23:13:27 knight"

import os.path
import re

import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Scanner
import SCons.Tool
import SCons.Util

class ToolQtWarning(SCons.Warnings.Warning):
        pass

class GeneratedMocFileNotIncluded(ToolQtWarning):
        pass

class QtdirNotFound(ToolQtWarning):
        pass

SCons.Warnings.enableWarningClass(ToolQtWarning)

qrcinclude_re = re.compile(r'<file>([^<]*)</file>', re.M)

def transformToWinePath(path) :
        return os.popen('winepath -w "%s"'%path).read().strip().replace('\\','/')

header_extensions = [".h", ".hxx", ".hpp", ".hh"]
if SCons.Util.case_sensitive_suffixes('.h', '.H'):
        header_extensions.append('.H')
# TODO: The following two lines will work when integrated back to SCons
# TODO: Meanwhile the third line will do the work
#cplusplus = __import__('c++', globals(), locals(), [])
#cxx_suffixes = cplusplus.CXXSuffixes
cxx_suffixes = [".c", ".cxx", ".cpp", ".cc"]

def checkMocIncluded(target, source, env):
        moc = target[0]
        cpp = source[0]
        # looks like cpp.includes is cleared before the build stage :-(
        # not really sure about the path transformations (moc.cwd? cpp.cwd?) :-/
        path = SCons.Defaults.CScan.path_function(env, moc.cwd)
        includes = SCons.Defaults.CScan(cpp, env, path)
        if not moc in includes:
                SCons.Warnings.warn(
                        GeneratedMocFileNotIncluded,
                        "Generated moc file '%s' is not included by '%s'" %
                        (str(moc), str(cpp)))

def find_file(filename, paths, node_factory):
        for dir in paths:
                node = node_factory(filename, dir)
                if node.rexists():
                        return node
        return None

class _Automoc:
        """
        Callable class, which works as an emitter for Programs, SharedLibraries and
        StaticLibraries.
        """

        def __init__(self, objBuilderName):
                self.objBuilderName = objBuilderName

        def __call__(self, target, source, env):
                """
                Smart autoscan function. Gets the list of objects for the Program
                or Lib. Adds objects and builders for the special qt files.
                """
                try:
                        if int(env.subst('$QT5_AUTOSCAN')) == 0:
                                return target, source
                except ValueError:
                        pass
                try:
                        debug = int(env.subst('$QT5_DEBUG'))
                except ValueError:
                        debug = 0

                # some shortcuts used in the scanner
                splitext = SCons.Util.splitext
                objBuilder = getattr(env, self.objBuilderName)

                # some regular expressions:
                # Q_OBJECT detection
                q_object_search = re.compile(r'[^A-Za-z0-9]Q_OBJECT[^A-Za-z0-9]')
                # cxx and c comment 'eater'
                #comment = re.compile(r'(//.*)|(/\*(([^*])|(\*[^/]))*\*/)')
                # CW: something must be wrong with the regexp. See also bug #998222
                #        CURRENTLY THERE IS NO TEST CASE FOR THAT

                # The following is kind of hacky to get builders working properly (FIXME)
                objBuilderEnv = objBuilder.env
                objBuilder.env = env
                mocBuilderEnv = env.Moc5.env
                env.Moc5.env = env

                # make a deep copy for the result; MocH objects will be appended
                out_sources = source[:]

                for obj in source:
                        if isinstance(obj,basestring):  # big kludge!
                                print("scons: qt5: '%s' MAYBE USING AN OLD SCONS VERSION AND NOT CONVERTED TO 'File'. Discarded." % str(obj))
                                continue
                        if not obj.has_builder():
                                # binary obj file provided
                                if debug:
                                        print("scons: qt: '%s' seems to be a binary. Discarded." % str(obj))
                                continue
                        cpp = obj.sources[0]
                        if not splitext(str(cpp))[1] in cxx_suffixes:
                                if debug:
                                        print("scons: qt: '%s' is no cxx file. Discarded." % str(cpp))
                                # c or fortran source
                                continue
                        #cpp_contents = comment.sub('', cpp.get_contents())
                        try:
                                cpp_contents = cpp.get_contents()
                        except: continue # may be an still not generated source
                        h=None
                        for h_ext in header_extensions:
                                # try to find the header file in the corresponding source
                                # directory
                                hname = splitext(cpp.name)[0] + h_ext
                                h = find_file(hname, (cpp.get_dir(),), env.File)
                                if h:
                                        if debug:
                                                print("scons: qt: Scanning '%s' (header of '%s')" % (str(h), str(cpp)))
                                        #h_contents = comment.sub('', h.get_contents())
                                        h_contents = h.get_contents()
                                        break
                        if not h and debug:
                                print("scons: qt: no header for '%s'." % (str(cpp)))
                        if h and q_object_search.search(h_contents):
                                # h file with the Q_OBJECT macro found -> add moc_cpp
                                moc_cpp = env.Moc5(h)
                                moc_o = objBuilder(moc_cpp)
                                out_sources.append(moc_o)
                                #moc_cpp.target_scanner = SCons.Defaults.CScan
                                if debug:
                                        print("scons: qt: found Q_OBJECT macro in '%s', moc'ing to '%s'" % (str(h), str(moc_cpp)))
                        if cpp and q_object_search.search(cpp_contents):
                                # cpp file with Q_OBJECT macro found -> add moc
                                # (to be included in cpp)
                                moc = env.Moc5(cpp)
                                env.Ignore(moc, moc)
                                if debug:
                                        print("scons: qt: found Q_OBJECT macro in '%s', moc'ing to '%s'" % (str(cpp), str(moc)))
                                #moc.source_scanner = SCons.Defaults.CScan
                # restore the original env attributes (FIXME)
                objBuilder.env = objBuilderEnv
                env.Moc5.env = mocBuilderEnv

                return (target, out_sources)

AutomocShared = _Automoc('SharedObject')
AutomocStatic = _Automoc('StaticObject')

def _detect(env):
        """Not really safe, but fast method to detect the QT library"""
        try: return env['QTDIR']
        except KeyError: pass

        try: return os.environ['QTDIR']
        except KeyError: pass

        moc = env.WhereIs('moc-qt5') or env.WhereIs('moc5') or env.WhereIs('moc')
        if moc:
                QTDIR = os.path.dirname(os.path.dirname(moc))
#               SCons.Warnings.warn(
#                       QtdirNotFound,
#                       "QTDIR variable is not defined, using moc executable as a hint (QTDIR=%s)" % QTDIR)
                return QTDIR

        raise SCons.Errors.StopError(
                QtdirNotFound,
                "Could not detect Qt 5 installation")
        return None

def generate(env):
        """Add Builders and construction variables for qt to an Environment."""

        def locateQt5Command(env, command, qtdir) :
                suffixes = [
                        '-qt5',
                        '-qt5.exe',
                        '5',
                        '5.exe',
                        '',
                        '.exe',
                ]
                triedPaths = []
                for suffix in suffixes :
                        fullpath = os.path.join(qtdir,'bin',command + suffix)
                        if os.access(fullpath, os.X_OK) :
                                return fullpath
                        triedPaths.append(fullpath)

                fullpath = env.Detect([command+'-qt5', command+'5', command])
                if not (fullpath is None) : return fullpath

                raise Exception("Qt5 command '" + command + "' not found. Tried: " + ', '.join(triedPaths))


        CLVar = SCons.Util.CLVar
        Action = SCons.Action.Action
        Builder = SCons.Builder.Builder
        splitext = SCons.Util.splitext

        env['QTDIR']  = env.GetBuildPath(_detect(env))

        # TODO: 'Replace' should be 'SetDefault'
#       env.SetDefault(
        env.Replace(
                QTDIR  = _detect(env),
                QT5_BINPATH = os.path.join('$QTDIR', 'bin'),
                QT5_LIBPATH = os.path.join('$QTDIR', 'lib'),
                # TODO: This is not reliable to QTDIR value changes but needed in order to support '-qt5' variants
                QT5_MOC = locateQt5Command(env,'moc', env['QTDIR']),
                QT5_UIC = locateQt5Command(env,'uic', env['QTDIR']),
                QT5_RCC = locateQt5Command(env,'rcc', env['QTDIR']),

                QT5_AUTOSCAN = 1, # Should the qt tool try to figure out, which sources are to be moc'ed?

                # Some QT specific flags. I don't expect someone wants to
                # manipulate those ...
                QT5_UICFLAGS = CLVar(''),
                QT5_MOCFROMHFLAGS = CLVar(''),
                QT5_MOCFROMCXXFLAGS = CLVar('-i'),
                QT5_QRCFLAGS = '',

                # suffixes/prefixes for the headers / sources to generate
                QT5_UISUFFIX = '.ui',
                QT5_UICDECLPREFIX = 'ui_',
                QT5_UICDECLSUFFIX = '.h',
                QT5_MOCINCPREFIX = '-I',
                QT5_MOCHPREFIX = 'moc_',
                QT5_MOCHSUFFIX = '$CXXFILESUFFIX',
                QT5_MOCCXXPREFIX = '',
                QT5_MOCCXXSUFFIX = '.moc',
                QT5_QRCSUFFIX = '.qrc',
                QT5_QRCCXXSUFFIX = '$CXXFILESUFFIX',
                QT5_QRCCXXPREFIX = 'qrc_',
                QT5_MOCCPPPATH = [],
                QT5_MOCINCFLAGS = '$( ${_concat(QT5_MOCINCPREFIX, QT5_MOCCPPPATH, INCSUFFIX, __env__, RDirs)} $)',

                # Commands for the qt support ...
                QT5_UICCOM = '"$QT5_UIC" $QT5_UICFLAGS -o $TARGET $SOURCE',
                QT5_MOCFROMHCOM = '"$QT5_MOC" $QT5_MOCFROMHFLAGS $QT5_MOCINCFLAGS -o $TARGET $SOURCE',
                QT5_MOCFROMCXXCOM = [
                        '$QT5_MOC $QT5_MOCFROMCXXFLAGS $QT5_MOCINCFLAGS -o $TARGET $SOURCE',
                        Action(checkMocIncluded,None)],
                QT5_RCCCOM = '"$QT5_RCC" $QT5_QRCFLAGS $SOURCE -o $TARGET',
        )

        # Resource builder
        def scanResources(node, env, path, arg):
                # I've being careful on providing names relative to the qrc file
                # If that was not needed that code could be simplified a lot
                def recursiveFiles(basepath, path) :
                        result = []
                        for item in os.listdir(os.path.join(basepath, path)) :
                                itemPath = os.path.join(path, item)
                                if os.path.isdir(os.path.join(basepath, itemPath)) :
                                        result += recursiveFiles(basepath, itemPath)
                                else:
                                        result.append(itemPath)
                        return result
                contents = node.get_contents()
                includes = qrcinclude_re.findall(contents)
                qrcpath = os.path.dirname(node.path)
                dirs = [included for included in includes if os.path.isdir(os.path.join(qrcpath,included))]
                # dirs need to include files recursively
                for dir in dirs :
                        includes.remove(dir)
                        includes+=recursiveFiles(qrcpath,dir)
                return includes
        qrcscanner = SCons.Scanner.Scanner(name = 'qrcfile',
                function = scanResources,
                argument = None,
                skeys = ['.qrc'])
        qrcbuilder = Builder(
                action = SCons.Action.Action('$QT5_RCCCOM', '$QT5_RCCCOMSTR'),
                source_scanner = qrcscanner,
                src_suffix = '$QT5_QRCSUFFIX',
                suffix = '$QT5_QRCCXXSUFFIX',
                prefix = '$QT5_QRCCXXPREFIX',
                single_source = True
                )
        env.Append( BUILDERS = { 'Qrc': qrcbuilder } )

        # Interface builder
        uic5builder = Builder(
                action = SCons.Action.Action('$QT5_UICCOM', '$QT5_UICCOMSTR'),
                src_suffix='$QT5_UISUFFIX',
                suffix='$QT5_UICDECLSUFFIX',
                prefix='$QT5_UICDECLPREFIX',
                single_source = True
                #TODO: Consider the uiscanner on new scons version
                )
        env['BUILDERS']['Uic5'] = uic5builder

        # Metaobject builder
        mocBld = Builder(action={}, prefix={}, suffix={})
        for h in header_extensions:
                act = SCons.Action.Action('$QT5_MOCFROMHCOM', '$QT5_MOCFROMHCOMSTR')
                mocBld.add_action(h, act)
                mocBld.prefix[h] = '$QT5_MOCHPREFIX'
                mocBld.suffix[h] = '$QT5_MOCHSUFFIX'
        for cxx in cxx_suffixes:
                act = SCons.Action.Action('$QT5_MOCFROMCXXCOM', '$QT5_MOCFROMCXXCOMSTR')
                mocBld.add_action(cxx, act)
                mocBld.prefix[cxx] = '$QT5_MOCCXXPREFIX'
                mocBld.suffix[cxx] = '$QT5_MOCCXXSUFFIX'
        env['BUILDERS']['Moc5'] = mocBld

        # er... no idea what that was for
        static_obj, shared_obj = SCons.Tool.createObjBuilders(env)
        static_obj.src_builder.append('Uic5')
        shared_obj.src_builder.append('Uic5')

        # We use the emitters of Program / StaticLibrary / SharedLibrary
        # to scan for moc'able files
        # We can't refer to the builders directly, we have to fetch them
        # as Environment attributes because that sets them up to be called
        # correctly later by our emitter.
        env.AppendUnique(PROGEMITTER =[AutomocStatic],
                                         SHLIBEMITTER=[AutomocShared],
                                         LIBEMITTER  =[AutomocStatic],
                                        )

        # TODO: Does dbusxml2cpp need an adapter
        env.AddMethod(enable_modules, "EnableQt5Modules")

def enable_modules(self, modules, debug=False, crosscompiling=False) :
        import sys

        validModules = [
                'QtCore',
                'QtGui',
                'QtWidgets',
                'QtOpenGL',
                'Qt3Support',
                'QtAssistant', # deprecated
                'QtAssistantClient',
                'QtScript',
                'QtDBus',
                'QtSql',
                'QtSvg',
                # The next modules have not been tested yet so, please
                # maybe they require additional work on non Linux platforms
                'QtNetwork',
                'QtTest',
                'QtXml',
                'QtXmlPatterns',
                'QtUiTools',
                'QtDesigner',
                'QtDesignerComponents',
                'QtWebKit',
                'QtHelp',
                'QtScript',
                'QtScriptTools',
                'QtMultimedia',
                ]
        pclessModules = [
# in qt <= 4.3 designer and designerComponents are pcless, on qt5.4 they are not, so removed.
#               'QtDesigner',
#               'QtDesignerComponents',
        ]
        staticModules = [
                'QtUiTools',
        ]
        invalidModules=[]
        for module in modules:
                if module not in validModules :
                        invalidModules.append(module)
        if invalidModules :
                raise Exception("Modules %s are not Qt5 modules. Valid Qt5 modules are: %s"% (
                        str(invalidModules),str(validModules)))

        moduleDefines = {
                'QtScript'   : ['QT_SCRIPT_LIB'],
                'QtSvg'      : ['QT_SVG_LIB'],
                'Qt3Support' : ['QT_QT3SUPPORT_LIB','QT3_SUPPORT'],
                'QtSql'      : ['QT_SQL_LIB'],
                'QtXml'      : ['QT_XML_LIB'],
                'QtOpenGL'   : ['QT_OPENGL_LIB'],
                'QtGui'      : ['QT_GUI_LIB'],
                'QtWidgets'  : ['QT_WIDGETS_LIB'],
                'QtNetwork'  : ['QT_NETWORK_LIB'],
                'QtCore'     : ['QT_CORE_LIB'],
        }
        for module in modules :
                try : self.AppendUnique(CPPDEFINES=moduleDefines[module])
                except: pass
        debugSuffix = ''
        if (sys.platform=='darwin' or sys.platform.startswith('linux2')) and not crosscompiling :
                if debug : debugSuffix = '_debug'
                for module in modules :
                        if module not in pclessModules : continue
                        self.AppendUnique(LIBS=[module+debugSuffix])
                        self.AppendUnique(LIBPATH=[os.path.join("$QTDIR","lib")])
                        self.AppendUnique(CPPPATH=[os.path.join("$QTDIR","include","qt5")])
                        self.AppendUnique(CPPPATH=[os.path.join("$QTDIR","include","qt5",module)])
                pcmodules = [module+debugSuffix for module in modules if module not in pclessModules ]
                if 'QtDBus' in pcmodules:
                        self.AppendUnique(CPPPATH=[os.path.join("$QTDIR","include","qt5","QtDBus")])
                if "QtAssistant" in pcmodules:
                        self.AppendUnique(CPPPATH=[os.path.join("$QTDIR","include","qt5","QtAssistant")])
                        pcmodules.remove("QtAssistant")
                        pcmodules.append("QtAssistantClient")
                if sys.platform.startswith('linux'):
                        self.ParseConfig('pkg-config %s --libs --cflags'% ' '.join(pcmodules))
                elif sys.platform == 'darwin':
                        for module in pcmodules:
                                #self.AppendUnique(CPPPATH="$QTDIR/frameworks/%s.framework/Versions/5/Headers" % module)
                                self.Append(LINKFLAGS=['-framework', module])
                self["QT5_MOCCPPPATH"] = self["CPPPATH"]
                return
        if sys.platform == "win32" or crosscompiling :
                modules_libs = [m[:2]+'5'+m[2:] for m in modules]
                if crosscompiling:
                        transformedQtdir = transformToWinePath(self['QTDIR'])
                        self['QT5_MOC'] = "QTDIR=%s %s"%( transformedQtdir, self['QT5_MOC'])
                self.AppendUnique(CPPPATH=[os.path.join("$QTDIR","include")])
                self.AppendUnique(LIBS=[lib for lib in modules_libs if lib not in staticModules])
                self.PrependUnique(LIBS=[lib for lib in modules_libs if lib in staticModules])
                if 'QtOpenGL' in modules:
                        self.AppendUnique(LIBS=['opengl32'])
                self.AppendUnique(CPPPATH=[ '$QTDIR/include/'])
                self.AppendUnique(CPPPATH=[ '$QTDIR/include/'+module for module in modules])
                if crosscompiling :
                        self["QT5_MOCCPPPATH"] = [
                                path.replace('$QTDIR', transformedQtdir)
                                        for path in self['CPPPATH'] ]
                else :
                        self["QT5_MOCCPPPATH"] = self["CPPPATH"]
                self.AppendUnique(LIBPATH=[os.path.join('$QTDIR','lib')])
                return

def exists(env):
        return _detect(env)
