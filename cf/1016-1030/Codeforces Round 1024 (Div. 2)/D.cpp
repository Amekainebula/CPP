#include <bits/stdc++.h>
//#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int maxn = 2e6 + 10;
int ans = 0;
void msort(int b, int e, vi &a) // 归并排序
{
    vi c(e - b + 10);
    if (b == e)
        return;
    int mid = (b + e) / 2, i = b, j = mid + 1, k = b;
    msort(b, mid, a), msort(mid + 1, e, a);
    while (i <= mid && j <= e)
        if (a[i] <= a[j])
            c[k++] = a[i++];
        else
            c[k++] = a[j++], ans += mid - i + 1; // 统计答案
    while (i <= mid)
        c[k++] = a[i++];
    while (j <= e)
        c[k++] = a[j++];
    for (int l = b; l <= e; l++)
        a[l] = c[l];
}
void Murasame()
{
    int n;
    cin >> n;
    vi b, c;
    vi a(maxn + 1);
    b.pb(0);
    c.pb(0);
    ff(i, 1, n)
    {
        int x;
        cin >> x;
        if (i % 2 == 1)
        {
            b.pb(x);
        }
        else
        {
            c.pb(x);
        }
    }
    ans = 0;
    msort(1, n / 2 + n % 2, b);
    int t1 = ans;
    ans = 0;
    msort(1, n / 2, c);
    int t2 = ans;
    // cout << t1 << " " << t2 << endl;
    bool ok = (t1 % 2 != t2 % 2);
    sort(all1(b)), sort(all1(c));
    int c1 = 1, c2 = 1;
    ff(i, 1, n)
    {
        if (i % 2 == 1)
        {
            a[i] = b[c1];
            c1++;
        }
        else
        {
            a[i] = c[c2];
            c2++;
        }
    }
    if (ok)
        swap(a[n - 2], a[n]);
    ff(i, 1, n) cout << a[i] << " ";
    cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}