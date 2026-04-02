#include <bits/stdc++.h>
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define vc vector
#define vi vector<int>
#define endl '\n'
using namespace std;
vc<string> ans;
vc<vc<int>> a(4, vi(4)), b(4, vi(4)), c(4, vi(4));
void debug(vc<vc<int>> t)
{
    ff(i, 1, 3) ff(j, 1, 3) cout << t[i][j] << " \n"[j == 3];
}
void C(int x) // 第x列互换
{
    ans.pb("C" + to_string(x));
    ff(i, 1, 3)
    {
        swap(a[i][x], b[i][x]);
    }
}
void AL() // 将 A 矩阵向左旋转 90 度
{
    ans.pb("AL");
    c[2][2] = a[2][2];
    ff(i, 1, 3)
    {
        c[1][i] = a[i][3];
        c[i][3] = a[3][3 - i + 1];
        c[3][3 - i + 1] = a[3 - i + 1][1];
        c[3 - i + 1][1] = a[1][i];
    }
    a = c;
}
void BL() // 将 B 矩阵向左旋转 90 度
{
    ans.pb("BL");
    c[2][2] = b[2][2];
    ff(i, 1, 3)
    {
        c[1][i] = b[i][3];
        c[i][3] = b[3][3 - i + 1];
        c[3][3 - i + 1] = b[3 - i + 1][1];
        c[3 - i + 1][1] = b[1][i];
    }
    b = c;
}
void Murasame()
{
    ff(i, 1, 3) ff(j, 1, 3)
    {
        char c;
        cin >> c;
        a[i][j] = c - '0';
    }
    ff(i, 1, 3) ff(j, 1, 3)
    {
        char c;
        cin >> c;
        b[i][j] = c - '0';
    }
    if (a[2][2] == 1) // 中间不一样，直接交换第二列
    {
        C(2);
    }
    while (a[1][2] || a[2][1] || a[3][2] || a[2][3]) // 每条边中间数不对
    {
        while (!a[2][1])
        {
            AL();
        }
        while (b[2][1])
        {
            BL();
        }
        C(1);
    }
    while (a[1][1] || a[1][3] || a[3][1] || a[3][3]) // 边角不对
    {
        while (!a[1][1])
        {
            AL();
        }
        while (b[1][1])
        {
            BL();
        }
        AL();
        C(1);
        AL();
        C(1);
        AL();
        AL();
        AL();
        C(1);
    }
    cout << ans.size() << endl;
    for (auto x : ans)
        cout << x << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}