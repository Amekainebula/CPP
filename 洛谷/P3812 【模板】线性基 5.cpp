#include <bits/stdc++.h>
#define int long long
using namespace std;
long long n, k;
vector<long long> p(105);
const int bit = 51;
void ins(vector<long long> &p)
{
    for (int i = bit; i >= 0; i--)
    {
        for (int j = k; j < n; j++)
            if ((p[j] >> i) & 1)
            {
                swap(p[j], p[k]);
                break;
            }

        if (((p[k] >> i) & 1) == 0)
            continue;
        for (int j = 0; j < n; j++)
            if (j != k && ((p[j] >> i) & 1))
                p[j] ^= p[k];
        k++;
        if (k == n)
            break;
    }
}

long long ans = 0;
signed main()
{
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        cin >> p[i];
    }
    ins(p);
    for (int i = k; i >= 0; i--)
    {
        ans ^= p[i];
    }
    cout << ans << '\n';
    return 0;
}
